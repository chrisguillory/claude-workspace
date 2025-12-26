/**
 * Safe JavaScript execution wrapper with comprehensive serialization.
 * Handles all JS types that JSON.stringify can't serialize.
 *
 * This is an async IIFE that takes user code as a parameter.
 * Used by execute_javascript() via build_execute_javascript_async_script().
 *
 * Arguments:
 *   __userCode: JavaScript code string to evaluate
 *
 * Returns:
 *   Promise resolving to { success, result, result_type, error?, note? }
 */
(async function(__userCode) {
    // Helper to safely serialize any JavaScript value to JSON-compatible format
    function safeSerialize(value) {
        // Handle primitives and special types before JSON.stringify
        if (value === undefined) {
            return { success: true, result: null, result_type: 'undefined' };
        }
        if (value === null) {
            return { success: true, result: null, result_type: 'null' };
        }
        if (typeof value === 'function') {
            return { success: true, result: null, result_type: 'function', note: 'Functions cannot be serialized' };
        }
        if (typeof value === 'symbol') {
            return { success: true, result: value.toString(), result_type: 'symbol' };
        }
        if (typeof value === 'bigint') {
            return { success: true, result: value.toString(), result_type: 'bigint' };
        }
        // Handle special number values that JSON.stringify converts to null
        // Return string representations for AI visibility (matches CiC behavior)
        if (typeof value === 'number') {
            if (Number.isNaN(value)) {
                return { success: true, result: 'NaN', result_type: 'number', note: 'Value is NaN (not serializable to JSON)' };
            }
            if (!Number.isFinite(value)) {
                const repr = value > 0 ? 'Infinity' : '-Infinity';
                return { success: true, result: repr, result_type: 'number', note: `Value is ${repr} (not serializable to JSON)` };
            }
            if (Object.is(value, -0)) {
                return { success: true, result: '-0', result_type: 'number', note: 'Value is negative zero (-0)' };
            }
        }
        // DOM nodes cannot be serialized
        if (typeof Node !== 'undefined' && value instanceof Node) {
            return { success: true, result: null, result_type: 'unserializable', note: 'DOM nodes cannot be serialized' };
        }
        // Window object check (common mistake)
        if (typeof Window !== 'undefined' && value instanceof Window) {
            return { success: true, result: null, result_type: 'unserializable', note: 'Window object cannot be serialized' };
        }
        // RegExp converts to string representation
        if (value instanceof RegExp) {
            return { success: true, result: value.toString(), result_type: 'string' };
        }
        // Date converts to ISO string
        if (value instanceof Date) {
            return { success: true, result: value.toISOString(), result_type: 'string' };
        }
        // Map converts to object
        if (value instanceof Map) {
            return safeSerialize(Object.fromEntries(value));
        }
        // Set converts to array
        if (value instanceof Set) {
            return safeSerialize([...value]);
        }
        // WeakMap and WeakSet cannot be serialized - entries are non-enumerable by design
        // This is an ECMAScript specification constraint to prevent observable non-determinism from GC
        if (value instanceof WeakMap) {
            return { success: true, result: null, result_type: 'unserializable', note: 'WeakMap entries cannot be enumerated or serialized. WeakMap is designed for internal object metadata and deliberately prevents access to its contents.' };
        }
        if (value instanceof WeakSet) {
            return { success: true, result: null, result_type: 'unserializable', note: 'WeakSet entries cannot be enumerated or serialized. WeakSet is designed for tracking object membership and deliberately prevents access to its contents.' };
        }
        // Error objects - extract message, name, stack (would otherwise serialize to {})
        if (value instanceof Error) {
            return {
                success: true,
                result: { name: value.name, message: value.message, stack: value.stack || null },
                result_type: 'error'
            };
        }
        // ArrayBuffer cannot be directly serialized
        if (value instanceof ArrayBuffer) {
            return { success: true, result: null, result_type: 'unserializable', note: 'ArrayBuffer cannot be directly serialized. Use TypedArray (e.g., new Uint8Array(buffer)) to access contents.' };
        }
        // Blob contents require async reading
        if (typeof Blob !== 'undefined' && value instanceof Blob) {
            return { success: true, result: null, result_type: 'unserializable', note: 'Blob contents require async reading via blob.text() or blob.arrayBuffer().' };
        }
        // Generator objects cannot be serialized (have next() and Symbol.iterator)
        if (value && typeof value.next === 'function' && typeof value[Symbol.iterator] === 'function') {
            return { success: true, result: null, result_type: 'unserializable', note: 'Generator state cannot be serialized. Consume the generator and return the values instead.' };
        }

        // For objects and arrays, use JSON.stringify with circular reference detection
        const seen = new WeakSet();
        try {
            const serialized = JSON.stringify(value, function(key, val) {
                // Handle BigInt in nested objects
                if (typeof val === 'bigint') {
                    return val.toString();
                }
                // Handle special number values that JSON.stringify converts to null
                if (typeof val === 'number') {
                    if (Number.isNaN(val)) return '[NaN]';
                    if (!Number.isFinite(val)) return val > 0 ? '[Infinity]' : '[-Infinity]';
                    if (Object.is(val, -0)) return '[-0]';
                }
                // Detect circular references
                if (typeof val === 'object' && val !== null) {
                    if (seen.has(val)) {
                        return '[Circular Reference]';
                    }
                    seen.add(val);
                }
                return val;
            });

            // Parse back to get clean object (removes undefined values, etc.)
            const result = JSON.parse(serialized);
            const resultType = Array.isArray(value) ? 'array' : typeof value;
            return { success: true, result: result, result_type: resultType };
        } catch (e) {
            // JSON.stringify failed (shouldn't happen after our checks, but be safe)
            return { success: true, result: null, result_type: 'unserializable', note: e.message };
        }
    }

    try {
        // Try expression form first (most common case)
        // This handles: 1 + 1, document.title, Promise.resolve(42), () => value
        let result;

        try {
            const exprFn = new Function('return (' + __userCode + ')');
            result = exprFn();
        } catch (parseErr) {
            if (parseErr instanceof SyntaxError) {
                // Expression parsing failed, try as statement block
                // This handles: throw new Error(), if/for/while, var x = 1
                const stmtFn = new Function(__userCode);
                result = stmtFn();  // undefined for statements without return
            } else {
                throw parseErr;  // Re-throw non-syntax errors
            }
        }

        // Auto-await if result is thenable (Promise or Promise-like)
        // Use typeof check instead of instanceof for cross-context compatibility
        if (result && typeof result.then === 'function') {
            result = await result;
        }

        return safeSerialize(result);
    } catch (e) {
        return {
            success: false,
            result: null,
            result_type: 'unserializable',
            error: e.message || String(e),
            error_stack: e.stack || null,
            error_type: 'execution'
        };
    }
})
