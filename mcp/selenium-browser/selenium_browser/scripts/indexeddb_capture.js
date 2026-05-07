// noinspection JSAnnotator

/**
 * IndexedDB Capture Script
 *
 * Enumerates all IndexedDB databases for the current origin and exports
 * their complete contents in Playwright-compatible storageState format.
 *
 * Uses indexedDB.databases() (Chrome 71+) and getAll/getAllKeys (IndexedDB v2)
 * for efficient bulk retrieval.
 *
 * Complex types are serialized with __type markers for reconstruction:
 * - Date: { __type: 'Date', __value: ISO string }
 * - Map: { __type: 'Map', __value: [[key, value], ...] }
 * - Set: { __type: 'Set', __value: [item, ...] }
 * - ArrayBuffer: { __type: 'ArrayBuffer', __value: [byte, ...] }
 * - TypedArrays: { __type: 'Uint8Array', __value: [byte, ...] }
 * - Blob/File: { __type: 'Blob', __size, __mimeType, __note: 'Content not captured' }
 * - CryptoKey (extractable):
 *     { __type: 'CryptoKey', __format: 'jwk', __value: <jwk>,
 *       __algorithm, __usages, __extractable: true, __keyType }
 * - CryptoKey (non-extractable):
 *     { __type: 'CryptoKey', __extractable: false,
 *       __algorithm, __usages, __keyType, __exportError: <reason> }
 *     Restoring a non-extractable CryptoKey from this metadata-only marker
 *     requires file-level access (copy IndexedDB LevelDB folder for the
 *     origin into the target profile dir before browser launch). The JS
 *     API has no way to extract bytes from non-extractable keys.
 *
 * Returns: Promise<Array<{
 *   databaseName: string,
 *   version: number,
 *   objectStores: Array<{
 *     name: string,
 *     keyPath: string | string[] | null,
 *     autoIncrement: boolean,
 *     indexes: Array<{ name, keyPath, unique, multiEntry }>,
 *     records: Array<{ key, value }>
 *   }>
 * }>>
 */

return new Promise(async (resolve, reject) => {
    try {
        // Check if indexedDB.databases() is available (Chrome 71+)
        if (!indexedDB || !indexedDB.databases) {
            resolve([]);  // Graceful fallback for older browsers
            return;
        }

        const databases = await indexedDB.databases();
        const result = [];

        for (const dbInfo of databases) {
            const dbName = dbInfo.name;

            try {
                // Open the database
                const db = await new Promise((res, rej) => {
                    const request = indexedDB.open(dbName);
                    request.onsuccess = () => res(request.result);
                    request.onerror = () => rej(request.error);
                });

                const dbExport = {
                    databaseName: dbName,
                    version: db.version,
                    objectStores: []
                };

                // Iterate through object stores
                for (const storeName of db.objectStoreNames) {
                    try {
                        const tx = db.transaction(storeName, 'readonly');
                        const store = tx.objectStore(storeName);

                        // Get object store metadata
                        const storeExport = {
                            name: storeName,
                            keyPath: store.keyPath,
                            autoIncrement: store.autoIncrement,
                            indexes: [],
                            records: []
                        };

                        // Get indexes
                        for (const indexName of store.indexNames) {
                            const index = store.index(indexName);
                            storeExport.indexes.push({
                                name: index.name,
                                keyPath: index.keyPath,
                                unique: index.unique,
                                multiEntry: index.multiEntry
                            });
                        }

                        // Get all records using getAll/getAllKeys (IndexedDB v2)
                        const [keys, values] = await Promise.all([
                            new Promise((res, rej) => {
                                const req = store.getAllKeys();
                                req.onsuccess = () => res(req.result);
                                req.onerror = () => rej(req.error);
                            }),
                            new Promise((res, rej) => {
                                const req = store.getAll();
                                req.onsuccess = () => res(req.result);
                                req.onerror = () => rej(req.error);
                            })
                        ]);

                        for (let i = 0; i < keys.length; i++) {
                            storeExport.records.push({
                                key: await serializeValue(keys[i]),
                                value: await serializeValue(values[i])
                            });
                        }

                        dbExport.objectStores.push(storeExport);
                    } catch (storeError) {
                        console.warn(`Error reading store ${storeName}:`, storeError);
                    }
                }

                db.close();
                result.push(dbExport);
            } catch (dbError) {
                console.warn(`Error reading database ${dbName}:`, dbError);
            }
        }

        resolve(result);
    } catch (error) {
        reject(error.message || String(error));
    }

    /**
     * Type-aware serialization for complex IndexedDB types.
     * Recursively serializes objects, arrays, and special types.
     * Async because CryptoKey export uses crypto.subtle.exportKey (Promise).
     */
    async function serializeValue(value) {
        // Primitives
        if (value === null || value === undefined) return null;
        if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
            return value;
        }

        // Date
        if (value instanceof Date) {
            return { __type: 'Date', __value: value.toISOString() };
        }

        // CryptoKey
        // Detected before generic object handling — Object.entries(cryptoKey) returns []
        // for these opaque host objects, which would otherwise silently destroy them.
        if (value instanceof CryptoKey) {
            const meta = {
                __type: 'CryptoKey',
                __algorithm: value.algorithm,
                __usages: value.usages,
                __extractable: value.extractable,
                __keyType: value.type,  // 'secret', 'public', 'private'
            };
            if (value.extractable) {
                try {
                    meta.__format = 'jwk';
                    meta.__value = await crypto.subtle.exportKey('jwk', value);
                } catch (e) {
                    meta.__exportError = e.name + ': ' + e.message;
                }
            } else {
                meta.__exportError = 'non-extractable: JS API cannot export bytes; use user_data_dir to preserve via LevelDB';
            }
            return meta;
        }

        // Map
        if (value instanceof Map) {
            const entries = [];
            for (const [k, v] of value.entries()) {
                entries.push([await serializeValue(k), await serializeValue(v)]);
            }
            return { __type: 'Map', __value: entries };
        }

        // Set
        if (value instanceof Set) {
            const items = [];
            for (const v of value) items.push(await serializeValue(v));
            return { __type: 'Set', __value: items };
        }

        // ArrayBuffer
        if (value instanceof ArrayBuffer) {
            return { __type: 'ArrayBuffer', __value: Array.from(new Uint8Array(value)) };
        }

        // TypedArrays (Uint8Array, Int32Array, etc.)
        if (ArrayBuffer.isView(value) && !(value instanceof DataView)) {
            return { __type: value.constructor.name, __value: Array.from(value) };
        }

        // Blob/File - capture metadata only (content too expensive)
        if (value instanceof Blob) {
            const result = {
                __type: value instanceof File ? 'File' : 'Blob',
                __size: value.size,
                __mimeType: value.type,
                __note: 'Content not captured'
            };
            if (value instanceof File) {
                result.__name = value.name;
            }
            return result;
        }

        // Arrays
        if (Array.isArray(value)) {
            const out = [];
            for (const v of value) out.push(await serializeValue(v));
            return out;
        }

        // Plain objects
        if (typeof value === 'object') {
            const result = {};
            for (const [k, v] of Object.entries(value)) {
                result[k] = await serializeValue(v);
            }
            return result;
        }

        // Fallback
        return String(value);
    }
});
