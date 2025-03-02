To authenticate requests coming from a **mobile app (React Native)** to **AWS AppSync** using **OIDC (OpenID Connect) and AWS Cognito**, follow these steps:

---

## **1. Setup AWS Cognito for Authentication**
### **Step 1: Create a Cognito User Pool**
1. Go to [AWS Cognito Console](https://console.aws.amazon.com/cognito/)
2. Click **"Create a User Pool"**
3. Configure:
   - **Sign-in options**: Email, Phone, or Username
   - **App Clients**: Create a new app client (without secret key for mobile apps)
   - **Token Expiry Settings**: Configure **ID Token** and **Access Token**
4. **Save the User Pool ID** and **App Client ID**

### **Step 2: Enable OpenID Connect (OIDC) Provider in Cognito**
If you're using a third-party **OIDC provider (e.g., Google, Okta, Auth0)**:
1. In **Cognito User Pool**, go to **Identity Providers**
2. Select **OpenID Connect (OIDC)**
3. Enter **Issuer URL**, **Client ID**, and **Client Secret** of your OIDC provider
4. Map user attributes (e.g., email, name)
5. Save **OIDC Provider ARN**

---

## **2. Configure AWS AppSync to Use Cognito Authentication**
1. Go to [AWS AppSync Console](https://console.aws.amazon.com/appsync/home)
2. Open your GraphQL API
3. Click **Settings → Authorization**
4. Choose **Amazon Cognito User Pool** and select the **User Pool ID**
5. Click **Save**

**For OIDC Authentication** (if using an external OIDC provider):
1. Click **Add Authorization Mode**
2. Select **OpenID Connect (OIDC)**
3. Enter the **Issuer URL** from your OIDC provider
4. Configure expiration time
5. Click **Save**

---

## **3. Implement Authentication in React Native with AWS Amplify**
Use **AWS Amplify** for **Cognito Authentication** in a **React Native App**.

### **Step 1: Install Amplify Packages**
```bash
npm install aws-amplify @aws-amplify/auth @aws-amplify/api-graphql
```

### **Step 2: Configure Amplify in Your App**
Modify `aws-exports.js` or configure it manually:

```javascript
import { Amplify } from 'aws-amplify';

Amplify.configure({
  Auth: {
    region: "us-east-1",
    userPoolId: "us-east-1_xxxxx",
    userPoolWebClientId: "xxxxxxxx",
    oauth: {
      domain: "your-auth-domain.auth.us-east-1.amazoncognito.com",
      scope: ["openid", "email", "profile"],
      redirectSignIn: "yourapp://callback/",
      redirectSignOut: "yourapp://signout/",
      responseType: "token"
    }
  }
});
```

### **Step 3: Sign in Users**
```javascript
import { Auth } from 'aws-amplify';

const signIn = async () => {
  try {
    const user = await Auth.signIn("username", "password");
    console.log("User signed in:", user);
  } catch (error) {
    console.log("Error signing in:", error);
  }
};
```

### **Step 4: Get JWT Token for AppSync Requests**
```javascript
const getJwtToken = async () => {
  const session = await Auth.currentSession();
  return session.getIdToken().getJwtToken();
};
```

---

## **4. Configure Apollo Client for AppSync Authentication**
In **React Native**, pass the **Cognito JWT Token** in the **Authorization Header**:

```javascript
import { ApolloClient, InMemoryCache, HttpLink } from '@apollo/client';
import { Auth } from 'aws-amplify';

const getAuthToken = async () => {
  const session = await Auth.currentSession();
  return session.getIdToken().getJwtToken();
};

const client = new ApolloClient({
  link: new HttpLink({
    uri: "https://your-appsync-api-id.appsync-api.us-east-1.amazonaws.com/graphql",
    headers: async () => ({
      Authorization: await getAuthToken(),
    }),
  }),
  cache: new InMemoryCache(),
});

export default client;
```

---

## **5. Query AppSync with Authentication**
Now, you can make **authenticated GraphQL requests**:

```javascript
import { gql, useQuery } from '@apollo/client';

const GET_EMPLOYEES = gql`
  query GetAllEmployees {
    getAllEmployees {
      id
      name
      email
    }
  }
`;

const EmployeeList = () => {
  const { loading, error, data } = useQuery(GET_EMPLOYEES);

  if (loading) return <Text>Loading...</Text>;
  if (error) return <Text>Error: {error.message}</Text>;

  return (
    <FlatList
      data={data.getAllEmployees}
      keyExtractor={(item) => item.id}
      renderItem={({ item }) => (
        <Text>{item.name} - {item.email}</Text>
      )}
    />
  );
};
```

---

## **6. Secure AppSync with Fine-Grained Access Control**
To restrict access based on **user roles**, use **Cognito Groups** and AppSync’s **Authorization Rules**:

1. **Create User Groups in Cognito** (e.g., `Admin`, `Employee`)
2. In **AppSync Schema**, add **@auth directives**:
```graphql
type Employee @model
  @auth(rules: [
    { allow: groups, groups: ["Admin"], operations: [create, update, delete] },
    { allow: owner, ownerField: "email", operations: [read] }
  ]) {
  id: ID!
  name: String!
  email: String! @primaryKey
  department: Department
}
```
3. Now, **Admins** can create/update/delete employees, but normal users can only read their own data.

---

## **7. Summary**
✅ **Set up AWS Cognito User Pool** for authentication  
✅ **Enable OIDC Provider (Google, Auth0, Okta) if needed**  
✅ **Integrate AppSync with Cognito for secure access**  
✅ **Use AWS Amplify for user authentication in React Native**  
✅ **Pass Cognito JWT token in Apollo Client for secure GraphQL requests**  
✅ **Use Cognito Groups & @auth directives for fine-grained access control**  

---

### **Next Steps**
🔹 **Enable Multi-Factor Authentication (MFA) in Cognito**  
🔹 **Use AWS Lambda Triggers to Customize Authentication Flow**  
🔹 **Implement Offline Data Sync using AWS AppSync & Amplify**  

Would you like a **step-by-step guide on using AWS CDK to deploy Cognito and AppSync together?**