# CS179 - GPU Programming Lab1

---

## Question 1: Common mistakes

### 1.1

Create an integer pointer, sets the value to which it points to 3, adds 2 and
print the said value. Given code:

```cpp
void test1() {
    int *a = 3;
    *a = *a + 2;
    printf("%d\n", *a);
}
```

_Answer_:

```cpp
void test1() {
    int *a; *a = 3;  // int *a = 3 makes pointer a points to address 3
    *a = *a + 2;
    printf("%d\n", *a);
}
```

### 1.2

Create two integer pointers and sets the values to which they point to 2
and 3 respectively. Given code:

```cpp
void test2() {
    int *a, b;
    a = (int *) malloc(sizeof(int));
    b = (int *) malloc(sizeof(int));
    
    if (!(a && b)) {
        printf("Out of memory\n");
        exit(-1);
    }

    *a = 2;
    *b = 3;
}
```

_Answer_:

```cpp
void test2() {
    int *a, *b;  // Add * before b to make it a pointer
    a = (int *) malloc(sizeof(int));
    b = (int *) malloc(sizeof(int));
    
    if (!(a && b)) {
        printf("Out of memory\n");
        exit(-1);
    }

    *a = 2;
    *b = 3;
}
```

### 1.3 

Allocates an array of 1000 integers, and for i = 0, ..., 999, sets the
i-th element to i. Given code:

```cpp
void test3() {
    int i, *a = (int *) malloc(1000);

    if (!a) {
        printf("Out of memory\n");
        exit(-1);
    }
    for (i = 0; i < 1000; i++)
        *(i + a) = i;
}
```

__Answer__:

```cpp
void test3() {
    int i, *a = (int *) malloc(1000*sizeof(int));  // int array size 

    if (!a) {
        printf("Out of memory\n");
        exit(-1);
    }
    for (i = 0; i < 1000; i++)
        *(i*sizeof(int) + a) = i;
}
```

### 1.4 

Creates a two-dimensional array of size 3x100, and sets element (1,1) 
(counting from 0) to 5. Given code:

```cpp
void test4() {
    int **a = (int **) malloc(3 * sizeof (int *));
    a[1][1] = 5;
}

__Answer__:

```cpp
void test4() {
    int **a = (int **) malloc(3 * sizeof (int *));
    for(int j = 0; j < 3; ++j) {
        a[j] = (int *) malloc(100 * sizeof(int));  // must allocate memory
    }
    a[1][1] = 5;
}
```

### 1.5

Sets the value pointed to by a to an input, checks if the value pointed to by a is 0, and prints a message if it is. Given code:

```cpp
void test5() {
    int *a = (int *) malloc(sizeof (int));
    scanf("%d", a);
    if (!a)
        printf("Value is 0\n");
}
```

__Answer__:

```cpp
void test5() {
    int *a = (int *) malloc(sizeof (int));
    scanf("%d", a);
    if (!*a)  // need dereference for value
        printf("Value is 0\n");
}
```

## Question 2: Parallelization 

### 2.1

Given an input signal x[n], suppose we have two output signals y_1[n] and 
y_2[n], given by the difference equations:
```
y_1[n] = x[n - 1] + x[n] + x[n + 1]
y_2[n] = y_2[n - 2] + y_2[n - 1] + x[n]
```

Which calculation do you expect will have an easier and faster implementation 
on the GPU, and why?

__Answer__:

The calculation described by `y_1[n]` will be much easier to implement on GPU because
each calculation only depends on the input. On the other hand, `y_2[n]` depends on 
`y_2[n-2]` and `y_2[n-1]`, which causes difficulty to implement on GPU.


### 2.2

In class, we discussed how the exponential moving average (EMA), in comparison to the simple moving average (SMA), is much less suited for parallelization on the GPU. 

Recall that the EMA is given by:

```
y[n] = c * x[n] + (1 - c) * y[n - 1]
```

Suppose that c is close to 1, and we only require an approximation to y[n]. How can we get this approximation in a way that is parallelizable? (Explain in words, optionally along with pseudocode or equations.)

Hint: If c is close to 1, then 1 - c is close to 0. If you expand the recurrence relation a bit, what happens to the contribution (to y[n]) of the terms y[n - k] as k increases?

__Answer__:

If `c` is close to 1, then we can estimate `y[n]` using the formula:

```
y[n] = c * x[n] + (1-c) * (c * x[n-1] + (1-c) * y[n-2]) 
```

In the formula above, the multiplicity of `y[n-2]` is `(1-c)^2`. When c is close to 1,
this term is close to zero. Therefore we can approximate `y[n]` as:

```
y[n] = c * x[n] + (1-c) * c * x[n-1]
```



