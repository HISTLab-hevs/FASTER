/**
 * auth.js — AuthService
 * 
 * A thin state wrapper around the global `api` client. 
 * Responsible for maintaining the in-memory `currentUser` session state 
 * and providing simplified authentication helpers for the UI.
 */

class AuthService {
    /**
     * Initializes the authentication service.
     */
    constructor() {
        /** 
         * The current authenticated user object. 
         * Null if the user is not authenticated.
         * @type {Object|null} 
         */
        this.currentUser = null;
    }

    /**
     * Authenticates the user and sets the local session state.
     * 
     * @param {string} identifier - The username or email address.
     * @param {string} password - The raw password string.
     * @returns {Promise<{success: boolean, user?: Object, message?: string}>} Authentication result.
     */
    async login(identifier, password) {
        try {
            const d = await api.login(identifier, password);
            this.currentUser = {
                username: d.username,
                email: d.email || '',
                role: d.role || 'user',
                is_admin: !!d.is_admin,
            };
            return { success: true, user: this.currentUser };
        } catch (e) {
            return {
                success: false,
                message: e.response?.data?.detail || 'Login failed',
            };
        }
    }

    /**
     * Logs the user out by clearing the API token and resetting local state.
     */
    logout() {
        api.clearToken();
        this.currentUser = null;
    }

    /**
     * Checks if the user has an active session token.
     * 
     * @returns {boolean} True if a token exists, false otherwise.
     */
    isAuthenticated() {
        return api.isAuthenticated();
    }
}

// Global instance mapping logic.
const auth = new AuthService();
