OIDC stands for **OpenID Connect**, which is an authentication protocol built on top of the **OAuth 2.0** framework. It allows clients (such as web applications, mobile apps, or APIs) to verify the identity of a user based on authentication performed by an external authorization server, as well as to obtain basic profile information about the user.

### Key Features of OIDC:
1. **User Authentication**: OIDC provides a way for users to log in and prove their identity.
2. **User Information**: It allows clients to access user profile data (e.g., name, email) through a **UserInfo endpoint**.
3. **ID Token**: Introduces the concept of an **ID Token**, a JSON Web Token (JWT) that contains information about the authenticated user.
4. **Built on OAuth 2.0**: OIDC inherits the security and delegation capabilities of OAuth 2.0 while focusing on user authentication.
5. **Interoperability**: Designed for cross-platform and cross-service authentication.

### OIDC Flow
1. **User Initiates Login**: The user attempts to log in to the client application.
2. **Authorization Request**: The client redirects the user to the OIDC authorization server.
3. **Authentication**: The authorization server authenticates the user.
4. **Token Issuance**: After successful authentication, the server issues:
   - **ID Token**: Contains user authentication details.
   - **Access Token**: For accessing protected resources.
5. **Token Verification**: The client validates the tokens and establishes a session.

### Common Use Cases
- Single Sign-On (SSO)
- Identity federation across platforms
- Secure user authentication for APIs

Popular OIDC providers include **Google**, **Microsoft Azure AD**, **Auth0**, **Okta**, and **Keycloak**.
