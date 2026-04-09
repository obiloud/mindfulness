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
- [ ] Create `POST /auth/register` endpoint:
  - [ ] Generate access_token (15m) + refresh_token (7d)
  - [ ] Store refresh_token in DB
  - [ ] Return both tokens in response
- [ ] Create `POST /auth/login` endpoint:
  - [ ] Generate access_token (15m) + refresh_token (7d)
  - [ ] Store refresh_token in DB
  - [ ] Return both tokens in response
- [ ] Create `POST /auth/refresh` endpoint:
  - [ ] Accept refresh_token (cookie or JSON body)
  - [ ] Validate token against DB
  - [ ] Check for reuse (used_at is not null)
  - [ ] If reused: revoke all user tokens (security breach)
  - [ ] Create new token pair
  - [ ] Delete old refresh token
  - [ ] Return new tokens
- [ ] Create `POST /auth/logout` endpoint:
  - [ ] Revoke refresh token from DB
  - [ ] Clear client-side tokens

### 1.5 Dependency Injection
- [ ] Create `get_current_user()` sub-dependency for protected routes
- [ ] Create `get_refresh_token()` sub-dependency for refresh endpoint
- [ ] Add CORS headers for cookie-based auth

### 1.6 Security Hardening
- [ ] Add `HttpOnly` cookie support for refresh tokens
- [ ] Add IP address and user_agent tracking to refresh tokens
- [ ] Implement token rotation validation
- [ ] Add rate limiting to refresh endpoint

## Phase 2: Frontend (Gleam / Lustre) - The Orchestrator

### 2.1 Schema Updates
- [ ] Add `refresh_token: String` to `AuthResponse` in `client/src/api.gleam`
- [ ] Add `refresh_token: String` to `Model` in `client/src/client.gleam`
- [ ] Add `is_refreshing: Bool` to `Model` for concurrency control

### 2.2 State Management
- [ ] Update `auth_init()` to load both tokens from localStorage
- [ ] Update `update_auth_state()` to handle refresh token storage
- [ ] Add `RefreshStarted` message type to `AuthMsg`
- [ ] Add `RefreshCompleted` message type to `AuthMsg`
- [ ] Add `LogoutRequired` message type to `AuthMsg`

### 2.3 HTTP Client Interceptor
- [ ] Create `send_authenticated_request()` helper function
- [ ] Implement 401 error handling:
  - [ ] Check `is_refreshing` flag
  - [ ] If not refreshing: trigger refresh
  - [ ] Queue original request
  - [ ] On refresh success: re-dispatch queued requests
  - [ ] On refresh failure: trigger logout
- [ ] Implement request queuing mechanism
- [ ] Implement retry logic for queued requests

### 2.4 Effect Handling
- [ ] Wrap refresh logic in `effect.from()` for async handling
- [ ] Handle refresh promise resolution in Lustre update cycle
- [ ] Implement concurrent refresh prevention

### 2.5 Logout Flow
- [ ] Update logout to clear both tokens from localStorage
- [ ] Update logout to clear queued requests
- [ ] Update logout to transition to `LoggedOut` state

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