# Function in C++

## Similarity to Java

Function in C++ are similar to methods in Java. However,
there is no indication of `public` or `private` for each
function. `public` and `private` properties are defined
in group.

## The main function

A C++ program begins execution in a function call `main`
with the following signature:

```cpp
int main() {
    return 0;
}
```

By convention, `main` should return 0 unless the program
encounters an error. Another note, unlike languages like
Java or C#, C++ has a _one-pass compiler_, which means
the compiler will report an error if a function has not
yet been declared when we try to use it (hence we have
prototype).

```cpp
/* A function prototype */
float foo(int boo);
```

## Digital Roots

Given a number, we get its digital root by keep taking sum of 
its digits until we get a single digit number, this single 
digit number is the given number's digital root.

```
135 -> 1+3+7=11 -> 1+1=2, droot is 2
``` 

## Think recursively

The factorial of `n` can be computed as:

```cpp
int factorial(int n) {
    if (n == 0) {
        return 1;
    } else {
        return n * factorial(n-1);
    }
}
```

Solving a problem with recursion requires two steps:

    1. Determine how to solve the problem for simple cases (base case).
    2. Determine how to break down larger cases into smaller instances (recursive step).

One way of computing the digital root of a number is 
to iterate through all of its digits:

```cpp
int sumOfDigitsOf(int n) {
    int result = 0;
    while (n != 0) {
        result += n % 10; 
        n /= 10;
    } 
    return result
}
```

Now, we can think recursively:

```cpp
int sumOfDigitsOf(int n) {
    if (n / 10 == 0) {
        return n;
    } else {
        return sumOfDigitsOf(n/10) + (n%10);
    }
}
```
