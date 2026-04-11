# Refresh Token Rotation Implementation Plan

## Phase 1: Backend (FastAPI) - The Authority

### 1.1 Database Schema & Models
- [x] Verify `RefreshToken` model exists in `backend/datamodels/database.py`
- [x] Verify `RefreshToken` Pydantic model exists in `backend/datamodels/migrations.py`
- [x] Ensure `users` table has `refresh_tokens` relationship
- [x] Ensure `refresh_tokens` table has proper indexes (user_id, expires_at)

### 1.2 Token Utility Functions
- [x] Implement `hash_token()` in `backend/datamodels/token.py`
- [x] Implement `create_token_pair()` to generate access + refresh tokens
- [x] Implement `validate_access_token()` for JWT decode
- [x] Implement `validate_refresh_token()` with DB lookup
- [x] Implement `revoke_refresh_token()` for single token revocation
- [x] Implement `revoke_user_refresh_tokens()` for security breach protocol
- [x] Implement `store_refresh_token()` with metadata (ip, user_agent)
- [x] Implement `get_refresh_token_by_id()` for lookup

### 1.3 JWT Configuration
- [x] Update `ACCESS_TOKEN_EXPIRE_MINUTES` to 15 in `backend/agent_a_chat/routes/authentication.py`
- [x] Create `create_refresh_token()` function with 7-day expiry
- [x] Create `create_token_pair()` function returning both tokens
- [x] Add `SECRET_KEY` environment variable requirement

### 1.4 Auth Endpoints
- [x] Create `POST /auth/register` endpoint:
  - [x] Generate access_token (15m) + refresh_token (7d)
  - [x] Store refresh_token in DB
  - [x] Return both tokens in response
- [x] Create `POST /auth/login` endpoint:
  - [x] Generate access_token (15m) + refresh_token (7d)
  - [x] Store refresh_token in DB
  - [x] Return both tokens in response
- [x] Create `POST /auth/refresh` endpoint:
  - [x] Accept refresh_token (cookie or JSON body)
  - [x] Validate token against DB
  - [x] Check for reuse (used_at is not null)
  - [x] If reused: revoke all user tokens (security breach)
  - [x] Create new token pair
  - [x] Delete old refresh token
  - [x] Return new tokens
- [x] Create `POST /auth/logout` endpoint:
  - [x] Revoke refresh token from DB
  - [x] Clear client-side tokens

### 1.5 Dependency Injection
- [x] Create `get_current_user()` sub-dependency for protected routes
- [x] Create `get_refresh_token()` sub-dependency for refresh endpoint
- [x] Add CORS headers for cookie-based auth

### 1.6 Security Hardening
- [x] Add `HttpOnly` cookie support for refresh tokens
- [x] Add IP address and user_agent tracking to refresh tokens
- [x] Implement token rotation validation
- [ ] Add rate limiting to refresh endpoint

## Phase 2: Frontend (Gleam / Lustre) - The Orchestrator

### 2.1 Schema Updates
- [x] Add `refresh_token: String` to `AuthResponse` in `client/src/api.gleam`
- [x] Add `refresh_token: String` to `Model` in `client/src/client.gleam`
- [x] Add `is_refreshing: Bool` to `Model` for concurrency control

### 2.2 State Management
- [x] Update `auth_init()` to load both tokens from localStorage
- [x] Update `update_auth_state()` to handle refresh token storage
- [x] Add `RefreshStarted` message type to `AuthMsg`
- [x] Add `RefreshCompleted` message type to `AuthMsg`
- [x] Add `LogoutRequired` message type to `AuthMsg`

### 2.3 HTTP Client Interceptor
- [x] Update `send_message()` function
- [x] Implement 401 error handling:
  - [x] Check `is_refreshing` flag
  - [x] If not refreshing: trigger refresh
  - [x] Queue original request
  - [x] On refresh success: re-dispatch queued requests
  - [x] On refresh failure: trigger logout
- [x] Implement request queuing mechanism
- [x] Implement retry logic for queued requests

### 2.4 Effect Handling
- [x] Wrap refresh logic in `effect.from()` for async handling
- [x] Handle refresh promise resolution in Lustre update cycle
- [x] Implement concurrent refresh prevention

### 2.5 Logout Flow
- [x] Update logout to clear both tokens from localStorage
- [x] Update logout to clear queued requests
- [x] Update logout to transition to `LoggedOut` state

### 2.6 UI Updates
- [ ] Add loading state during refresh
- [ ] Add error handling for refresh failures
- [ ] Add logout confirmation

## Phase 3: Edge Cases & Security

### 3.1 Concurrent Refresh Prevention
- [ ] Implement `is_refreshing` flag in Model
- [ ] Use `effect.all` or similar for concurrent request handling
- [ ] Ensure only one refresh call per 401 storm

### 3.2 Expired Refresh Token Handling
- [ ] If refresh endpoint returns 401/400: trigger logout
- [ ] Clear all local storage
- [ ] Transition to `LoggedOut` state

### 3.3 Security Breach Protocol
- [ ] If old refresh token is reused:
  - [ ] Revoke all tokens for that user
  - [ ] Trigger logout for all sessions
  - [ ] Log security event

### 3.4 Cookie-Based Auth (Optional)
- [ ] Configure HttpOnly cookies for refresh tokens
- [ ] Configure SameSite=Strict for refresh tokens
- [ ] Configure Secure flag for production

## Phase 4: Testing & Validation

### 4.1 Unit Tests
- [ ] Test token creation and validation
- [ ] Test refresh token rotation
- [ ] Test token revocation
- [ ] Test concurrent refresh prevention
- [ ] Test security breach protocol

### 4.2 Integration Tests
- [ ] Test full login flow
- [ ] Test refresh flow with 401
- [ ] Test logout flow
- [ ] Test expired token handling

### 4.3 Security Tests
- [ ] Test token reuse detection
- [ ] Test concurrent refresh attacks
- [ ] Test token expiration handling

## Implementation Order

1. **Phase 1.1-1.2**: Database & Token Utilities (Backend)
2. **Phase 1.3-1.6**: Auth Endpoints (Backend)
3. **Phase 2.1-2.6**: Frontend State & HTTP Client
4. **Phase 3**: Edge Cases & Security
5. **Phase 4**: Testing

## Dependencies

- `python-jose` for JWT handling
- `sqlalchemy` for ORM operations
- `bcrypt` or `hashlib` for token hashing
- `fastapi` CORS for cookie support

## Notes

- Access Token: 15 minutes (JWT)
- Refresh Token: 7 days (stored in DB)
- Token Rotation: Always issue new pair on refresh
- Security: Revoke all tokens on reuse detection
- Concurrency: Single refresh call per 401 storm