// noinspection JSAnnotator

/**
 * IndexedDB Restore Script
 *
 * Restores IndexedDB databases from Playwright-compatible storageState format.
 * Clears existing databases and recreates them with captured schema and data.
 *
 * Takes as input (arguments[0]): Array of database exports from indexeddb_capture.js
 *
 * Handles schema creation during onupgradeneeded and data insertion after ready.
 * Deserializes complex types using __type markers from capture.
 *
 * Async deserialization rule
 * --------------------------
 * IndexedDB transactions auto-commit when control returns to the event loop.
 * `await crypto.subtle.importKey()` (used by the CryptoKey case) yields control,
 * which would silently commit any open transaction and fail subsequent puts.
 * To work around this we PRE-DESERIALIZE all records before opening the
 * transaction, then run the puts synchronously inside a fresh transaction.
 *
 * Returns: Promise<{ success: boolean, databases_restored: number, records_restored: number, errors: string[] }>
 */

const indexedDBData = arguments[0];

return new Promise(async (resolve, reject) => {
    const results = {
        success: true,
        databases_restored: 0,
        records_restored: 0,
        errors: []
    };

    if (!indexedDBData || indexedDBData.length === 0) {
        resolve(results);
        return;
    }

    try {
        for (const dbExport of indexedDBData) {
            try {
                // Delete existing database first (clean slate)
                await new Promise((res) => {
                    const deleteRequest = indexedDB.deleteDatabase(dbExport.databaseName);
                    deleteRequest.onsuccess = () => res();
                    deleteRequest.onerror = () => res(); // Continue even if delete fails
                    deleteRequest.onblocked = () => {
                        console.warn(`Delete blocked for ${dbExport.databaseName}`);
                        setTimeout(() => res(), 1000); // Wait and continue
                    };
                });

                // Open database at specified version - this triggers onupgradeneeded
                const db = await new Promise((res, rej) => {
                    const openRequest = indexedDB.open(dbExport.databaseName, dbExport.version);

                    openRequest.onerror = () => rej(openRequest.error);

                    openRequest.onupgradeneeded = (event) => {
                        const db = event.target.result;

                        // Create all object stores from export
                        for (const storeExport of dbExport.objectStores) {
                            try {
                                // Create object store with proper configuration
                                const storeOptions = {};
                                if (storeExport.keyPath !== null) {
                                    storeOptions.keyPath = storeExport.keyPath;
                                }
                                if (storeExport.autoIncrement) {
                                    storeOptions.autoIncrement = true;
                                }

                                const objectStore = db.createObjectStore(storeExport.name, storeOptions);

                                // Create indexes
                                for (const indexExport of storeExport.indexes || []) {
                                    try {
                                        objectStore.createIndex(
                                            indexExport.name,
                                            indexExport.keyPath,
                                            {
                                                unique: indexExport.unique || false,
                                                multiEntry: indexExport.multiEntry || false
                                            }
                                        );
                                    } catch (indexError) {
                                        console.warn(`Error creating index ${indexExport.name}:`, indexError);
                                    }
                                }
                            } catch (storeError) {
                                console.warn(`Error creating object store ${storeExport.name}:`, storeError);
                                results.errors.push(`Failed to create store ${storeExport.name}: ${storeError.message}`);
                            }
                        }
                    };

                    openRequest.onsuccess = () => res(openRequest.result);
                });

                // Now insert all data
                for (const storeExport of dbExport.objectStores) {
                    try {
                        if (!db.objectStoreNames.contains(storeExport.name)) {
                            console.warn(`Store ${storeExport.name} not found after creation`);
                            continue;
                        }

                        // PRE-DESERIALIZE all records before opening the transaction.
                        // See "Async deserialization rule" in the file header.
                        const deserialized = [];
                        for (const record of storeExport.records || []) {
                            try {
                                deserialized.push({
                                    key: await deserializeValue(record.key),
                                    value: await deserializeValue(record.value),
                                });
                            } catch (deserializeError) {
                                console.warn('Error deserializing record:', deserializeError);
                                results.errors.push(
                                    `Failed to deserialize record in ${storeExport.name}: ${deserializeError.message}`
                                );
                            }
                        }

                        // Now run all puts synchronously inside a fresh transaction.
                        const transaction = db.transaction(storeExport.name, 'readwrite');
                        const objectStore = transaction.objectStore(storeExport.name);

                        for (const rec of deserialized) {
                            try {
                                // Use put() which works for both in-line and out-of-line keys
                                if (storeExport.keyPath === null) {
                                    // Out-of-line keys - provide key explicitly
                                    objectStore.put(rec.value, rec.key);
                                } else {
                                    // In-line keys - key is within the value
                                    objectStore.put(rec.value);
                                }

                                results.records_restored++;
                            } catch (recordError) {
                                console.warn(`Error inserting record:`, recordError);
                                results.errors.push(`Failed to insert record in ${storeExport.name}: ${recordError.message}`);
                            }
                        }

                        // Wait for transaction to complete
                        await new Promise((res, rej) => {
                            transaction.oncomplete = () => res();
                            transaction.onerror = () => rej(transaction.error);
                        });

                    } catch (storeError) {
                        console.warn(`Error restoring store ${storeExport.name}:`, storeError);
                        results.errors.push(`Failed to restore store ${storeExport.name}: ${storeError.message}`);
                    }
                }

                db.close();
                results.databases_restored++;

            } catch (dbError) {
                console.warn(`Error restoring database ${dbExport.databaseName}:`, dbError);
                results.errors.push(`Failed to restore database ${dbExport.databaseName}: ${dbError.message}`);
                results.success = false;
            }
        }

        resolve(results);
    } catch (error) {
        results.success = false;
        results.errors.push(error.message || String(error));
        resolve(results);
    }

    /**
     * Deserialize values with __type markers back to original types.
     * Reverse of serializeValue from indexeddb_capture.js.
     * Async because CryptoKey case uses crypto.subtle.importKey (Promise).
     */
    async function deserializeValue(value) {
        if (value === null || value === undefined) {
            return null;
        }

        // Check for type markers
        if (value && typeof value === 'object' && value.__type) {
            switch (value.__type) {
                case 'Date':
                    return new Date(value.__value);

                case 'CryptoKey':
                    // Extractable keys carry __value (the JWK) and round-trip cleanly.
                    if (value.__value && value.__format) {
                        return await crypto.subtle.importKey(
                            value.__format,
                            value.__value,
                            value.__algorithm,
                            value.__extractable,
                            value.__usages
                        );
                    }
                    // Non-extractable keys captured with metadata only cannot be
                    // reconstructed with original bytes via JS API. Return a marker
                    // so callers can detect the gap. To preserve identity, the caller
                    // must use the user_data_dir parameter on navigate to keep the
                    // browser's IndexedDB LevelDB folder intact.
                    return {
                        __type: 'CryptoKey',
                        __unrestored: true,
                        __reason: value.__exportError || 'non-extractable',
                        __algorithm: value.__algorithm,
                        __usages: value.__usages,
                        __extractable: value.__extractable,
                        __keyType: value.__keyType,
                    };

                case 'Map': {
                    const entries = [];
                    for (const [k, v] of value.__value) {
                        entries.push([await deserializeValue(k), await deserializeValue(v)]);
                    }
                    return new Map(entries);
                }

                case 'Set': {
                    const items = [];
                    for (const v of value.__value) {
                        items.push(await deserializeValue(v));
                    }
                    return new Set(items);
                }

                case 'ArrayBuffer':
                    return new Uint8Array(value.__value).buffer;

                case 'Uint8Array':
                case 'Int8Array':
                case 'Uint16Array':
                case 'Int16Array':
                case 'Uint32Array':
                case 'Int32Array':
                case 'Float32Array':
                case 'Float64Array': {
                    const TypedArrayConstructor = globalThis[value.__type];
                    return new TypedArrayConstructor(value.__value);
                }

                case 'Blob':
                case 'File':
                    // Cannot reconstruct Blob/File from metadata alone
                    // Return metadata object as-is
                    return value;

                default:
                    // Unknown type marker, return as-is
                    return value;
            }
        }

        // Arrays
        if (Array.isArray(value)) {
            const out = [];
            for (const v of value) {
                out.push(await deserializeValue(v));
            }
            return out;
        }

        // Plain objects
        if (typeof value === 'object') {
            const result = {};
            for (const [k, v] of Object.entries(value)) {
                result[k] = await deserializeValue(v);
            }
            return result;
        }

        // Primitives
        return value;
    }
});
