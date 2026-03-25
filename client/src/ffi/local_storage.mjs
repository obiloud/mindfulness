// client/src/ffi/local_storage.mjs

/**
 * A set of functions to manipulate localStorage
 * 
 * These functions provide a simple interface to the browser's localStorage API
 * with error handling and type safety
 */

export const get_item = (key) => {
    try {
        return window.localStorage.getItem(key);
    } catch (error) {
        console.error(`Error reading from localStorage: ${error.message}`);
        return null;
    }
};

export const set_item = (key, value) => {
    try {
        // Convert value to string before storing
        const stringValue = String(value);
        window.localStorage.setItem(key, stringValue);
        return true;
    } catch (error) {
        console.error(`Error writing to localStorage: ${error.message}`);
        return false;
    }
};

export const remove_item = (key) => {
    try {
        window.localStorage.removeItem(key);
        return true;
    } catch (error) {
        console.error(`Error removing from localStorage: ${error.message}`);
        return false;
    }
};

export const clear_storage = () => {
    try {
        window.localStorage.clear();
        return true;
    } catch (error) {
        console.error(`Error clearing localStorage: ${error.message}`);
        return false;
    }
};

export const has_item = (key) => {
    try {
        return window.localStorage.getItem(key) !== null;
    } catch (error) {
        console.error(`Error checking localStorage: ${error.message}`);
        return false;
    }
};

export const get_all_keys = () => {
    try {
        return [...window.localStorage.keys()].filter(key => key !== null);
    } catch (error) {
        console.error(`Error retrieving keys from localStorage: ${error.message}`);
        return [];
    }
};

export const get_all_items = () => {
    try {
        const keys = [...window.localStorage.keys()];
        const items = {};
        keys.forEach(key => {
            items[key] = window.localStorage.getItem(key);
        });
        return items;
    } catch (error) {
        console.error(`Error retrieving all items from localStorage: ${error.message}`);
        return {};
    }
};

// Optional: Add a function to get the size of localStorage
export const get_storage_size = () => {
    try {
        let size = 0;
        const keys = [...window.localStorage.keys()];
        keys.forEach(key => {
            const value = window.localStorage.getItem(key);
            if (value) {
                size += value.length;
            }
        });
        return size;
    } catch (error) {
        console.error(`Error getting storage size: ${error.message}`);
        return 0;
    }
};