In React, hooks are functions that let you use state and lifecycle features in functional components. Here are some of the most commonly used hooks:  

### 1. **useState**  
   - Manages state in functional components.  
   - Example:  
     ```jsx
     const [count, setCount] = useState(0);
     setCount(count + 1);
     ```

### 2. **useEffect**  
   - Handles side effects like fetching data, subscriptions, or manually changing the DOM.  
   - Example:  
     ```jsx
     useEffect(() => {
       console.log("Component mounted or updated");
     }, [count]);  // Runs when `count` changes
     ```

### 3. **useContext**  
   - Accesses values from React's Context API without prop drilling.  
   - Example:  
     ```jsx
     const theme = useContext(ThemeContext);
     ```

### 4. **useRef**  
   - Creates a reference to DOM elements or mutable values that don’t trigger re-renders.  
   - Example:  
     ```jsx
     const inputRef = useRef(null);
     useEffect(() => inputRef.current.focus(), []);
     ```

### 5. **useMemo**  
   - Optimizes performance by memoizing computed values.  
   - Example:  
     ```jsx
     const expensiveValue = useMemo(() => computeExpensiveValue(a, b), [a, b]);
     ```

### 6. **useCallback**  
   - Memoizes functions to prevent unnecessary re-renders.  
   - Example:  
     ```jsx
     const memoizedCallback = useCallback(() => doSomething(a, b), [a, b]);
     ```

### 7. **useReducer**  
   - Manages complex state logic, an alternative to `useState`.  
   - Example:  
     ```jsx
     const [state, dispatch] = useReducer(reducer, initialState);
     ```

### 8. **useLayoutEffect**  
   - Similar to `useEffect`, but runs synchronously after all DOM updates.  
   - Example:  
     ```jsx
     useLayoutEffect(() => {
       console.log("Runs after DOM mutations");
     }, []);
     ```

### 9. **useImperativeHandle**  
   - Customizes the instance value exposed by `useRef` when using `forwardRef`.  
   - Example:  
     ```jsx
     useImperativeHandle(ref, () => ({
       customMethod() {
         console.log("Exposed method");
       }
     }));
     ```

### 10. **useDebugValue**  
   - Provides a custom label in React DevTools for debugging custom hooks.  
   - Example:  
     ```jsx
     useDebugValue(value, v => `Formatted: ${v}`);
     ```

Would you like details on a specific hook?