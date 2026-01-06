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
                                key: serializeValue(keys[i]),
                                value: serializeValue(values[i])
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
     */
    function serializeValue(value) {
        // Primitives
        if (value === null || value === undefined) return null;
        if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
            return value;
        }

        // Date
        if (value instanceof Date) {
            return { __type: 'Date', __value: value.toISOString() };
        }

        // Map
        if (value instanceof Map) {
            return {
                __type: 'Map',
                __value: Array.from(value.entries()).map(([k, v]) => [serializeValue(k), serializeValue(v)])
            };
        }

        // Set
        if (value instanceof Set) {
            return { __type: 'Set', __value: Array.from(value).map(v => serializeValue(v)) };
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
            return value.map(v => serializeValue(v));
        }

        // Plain objects
        if (typeof value === 'object') {
            const result = {};
            for (const [k, v] of Object.entries(value)) {
                result[k] = serializeValue(v);
            }
            return result;
        }

        // Fallback
        return String(value);
    }
});