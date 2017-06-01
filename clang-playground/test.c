enum Enum1 {
  E1A,
  E1B,
  E1C=1,
  E1D
};

typedef struct B BAlias;
typedef BAlias Anotheralize;

struct A {
  BAlias *b;
};

typedef struct C CAlias;

typedef struct D* DPointer;

struct D;

struct F {} *var;
typedef struct E {} Node,*EPointer;

typedef struct D {
  CAlias *c;
} XX;

typedef struct C {
} CAlias;

typedef struct {
} *anp;

/* struct node { */
/* }; */

/* typedef struct node node; */
/* typedef struct node node; */
typedef struct {
  int a;
} node;

/* typedef struct node2 node2; */
typedef struct node2 {
  int a;
} node2;

/* typedef struct node Anotehrnode; */
/* struct node x; */


struct B {};

extern int b=0;
int a;

/* #include <sys/types.h> */

ssize_t (*original_write)(int, const void *, size_t);

hebi foo() {}


int main() {
  /* A *a; */
}
