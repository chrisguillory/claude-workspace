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
                await new Promise((res, rej) => {
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

                        const transaction = db.transaction(storeExport.name, 'readwrite');
                        const objectStore = transaction.objectStore(storeExport.name);

                        // Insert all records
                        for (const record of storeExport.records || []) {
                            try {
                                const deserializedValue = deserializeValue(record.value);

                                // Use put() which works for both in-line and out-of-line keys
                                if (storeExport.keyPath === null) {
                                    // Out-of-line keys - provide key explicitly
                                    objectStore.put(deserializedValue, record.key);
                                } else {
                                    // In-line keys - key is within the value
                                    objectStore.put(deserializedValue);
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
     * Reverse of serializeValue from indexeddb_capture.js
     */
    function deserializeValue(value) {
        if (value === null || value === undefined) {
            return null;
        }

        // Check for type markers
        if (value && typeof value === 'object' && value.__type) {
            switch (value.__type) {
                case 'Date':
                    return new Date(value.__value);

                case 'Map':
                    return new Map(value.__value.map(([k, v]) => [
                        deserializeValue(k),
                        deserializeValue(v)
                    ]));

                case 'Set':
                    return new Set(value.__value.map(v => deserializeValue(v)));

                case 'ArrayBuffer':
                    return new Uint8Array(value.__value).buffer;

                case 'Uint8Array':
                case 'Int8Array':
                case 'Uint16Array':
                case 'Int16Array':
                case 'Uint32Array':
                case 'Int32Array':
                case 'Float32Array':
                case 'Float64Array':
                    const TypedArrayConstructor = globalThis[value.__type];
                    return new TypedArrayConstructor(value.__value);

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
            return value.map(v => deserializeValue(v));
        }

        // Plain objects
        if (typeof value === 'object') {
            const result = {};
            for (const [k, v] of Object.entries(value)) {
                result[k] = deserializeValue(v);
            }
            return result;
        }

        // Primitives
        return value;
    }
});