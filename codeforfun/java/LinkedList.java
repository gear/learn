import java.util.*;
import java.io.*;

class LinkedList {
  public Node head;

  public static void main(String[] args) {
    LinkedList list = new LinkedList();
    list.head = new Node(0);
    list.head.AppendToEnd(new Node(2));
    list.head.AppendToEnd(new Node(3));
    list.head.AppendToEnd(new Node(1));
    list.head.AppendToEnd(new Node(2));
    list.head.AppendToEnd(new Node(2));
    list.head.AppendToEnd(new Node(2));
    list.head.AppendToEnd(new Node(2));
    list.head.AppendToEnd(new Node(2));
    list.head.AppendToEnd(new Node(1));
    System.out.println("First list:");
    list.head.PrintToEnd();
    LinkedList list2 = new LinkedList();
    list2.head = new Node(0);
    list2.head.AppendToEnd(new Node(3));
    list2.head.AppendToEnd(new Node(5));
    list2.head.AppendToEnd(new Node(1));
    list2.head.AppendToEnd(new Node(2));
    list2.head.AppendToEnd(new Node(2));
    list2.head.AppendToEnd(new Node(1));
    list2.head.AppendToEnd(new Node(2));
    list2.head.AppendToEnd(new Node(2));
    list2.head.AppendToEnd(new Node(1));
    System.out.println("Second list:");
    list2.head.PrintToEnd();
    list.Add(list2);
    System.out.println("Result:");
    list.head.PrintToEnd();
  }

  public void Add(LinkedList other) {
    Node seeker = other.head;
    Node adder = this.head;
    int c = 0;
    while (seeker != null || c > 0) {
      if (seeker == null)
        seeker = new Node(0);
      int acc = seeker.key + adder.key + c;
      if (acc > 10) {
        adder.key = acc % 10;
        c = acc / 10;
      } else {
        adder.key = acc;
        c = 0;
      }
      if (adder.next == null) {
        adder.next = new Node(0);
      }
      adder = adder.next;
      seeker = seeker.next;
    }
  }

  public void Partition(int val) {
    Node seeker = this.head;
    Node larger = null;
    Node smaller = null;;
    Node tmp;
    while (seeker != null) {
      if (seeker.key >= val) {
        if (larger != null) {
          larger.AppendToEnd(seeker);
        } else {
          larger = seeker;
        }
      } else {
        if (smaller != null) {
          smaller.AppendToEnd(seeker);
        } else {
          smaller = seeker;
        }
      }
      tmp = seeker.next;
      seeker.next = null;
      seeker = tmp;
    }
    if (smaller == null) {
      this.head = larger;
    } else {
      if (larger == null) {
        this.head = smaller;
      } else {
        smaller.AppendToEnd(larger);
        this.head = smaller;
      }
    }
  }
}

class Node {
  public Node next;
  public int key;
  public Node prev;

  public Node(int key) {
    this.key = key;
  }

  public void AppendToEnd(Node n) {
    Node seeker = this;
    while (seeker.next != null) {
      seeker = seeker.next;
    }
    seeker.next = n; 
  }

  public void PrintToEnd() {
    Node seeker = this;
    while (seeker.next != null) {
      System.out.print(seeker.key + " -> ");
      seeker = seeker.next;
    }
    System.out.println(seeker.key);
  }
}
