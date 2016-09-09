import java.io.*;

public class ThreeStack {
  public static void main(String[] args) {
     
  }
}

class Stack {
  public StackNode head;

  public void push(StackNode n) {
    if (this.head == null)
      this.head = new StackNode();
    this.head.next = n;
  }

  public StackNode pop() throws EmptyStackException {
    if (this.head == null) throw new EmptyStackException();
    StackNode ret = this.head;
    this.head = this.head.next;
    return ret;
  }

  public StackNode peek() {
    return this.head;
  }
}

class StackNode {
  public StackNode next;
  public int key;

  StackNode(int key) {
    this.key = key;
  }

  StackNode() {}

  StackNode(StackNode next) {
    this.next = next;
  }
}

class EmptyStackException extends Exception {
  public EmptyStackException() {
    super("Try to pop an empty stack.");
  }

  public EmptyStackException(String message) {
    super(message);
  }
}
