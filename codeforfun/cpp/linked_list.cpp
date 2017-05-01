#include <iostream>

struct Node {
  int data;
  struct Node* next;
}

int length(struct Node* node_ptr) {
  if (node_ptr == NULL)
    return 0;
  else
    return 1 + length(node_ptr->next);
}

/* Can return by reference */
int search(struct Node* head, int query_data) {
  /* Return the distance of the element to head */
  int dist = 0;
  struct Node* probe = head;
  while (probe != NULL) {
    if (probe->data == query_data)
      return dist;
    else {
      probe = probe->next;
      ++dist;
    }
  }
  return -1; /* Not found */
}

void push(struct Node** ref, int data) {
  struct Node* new_node = (struct Node*) malloc(sizeof(struct Node));
  new_node->data = data;
  new_node->next = *ref;
  *ref = new_node;
}

struct Node& get(struct Node* HEAD, int loc, bool reversed=false) {

}

int main() {
  struct Node* HEAD = NULL;
  push(&HEAD, 1); push(&HEAD, 2);
  push(&HEAD, 3); push(&HEAD, 4);
  push(&HEAD, 4); push(&HEAD, 5);
  push(&HEAD, 6); push(&HEAD, 7);
  /* Note: 7 -> 6 -> 5 -> 4 -> 4 -> 3 -> 2 -> 1 */

  int len = length(HEAD);
  printf("Length of list: %d (expected 8)\n", len);

  int query = 4;
  int loc = search(HEAD, query);
  printf("First instance of %d query: %d (expected 3) \n", query, loc);
  query = 8; loc = search(HEAD, query);
  printf("First instance of %d query: %d (expected -1) \n", query, loc);

}
