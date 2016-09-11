import java.util.*;
import java.io.*;

public class TopK {
  public static void main(String[] args) {
    Dequeue la = new Dequeue(5);
    la.push(1);
    for(int t : la.queue) {
      System.out.print(t + " ");
    }
    System.out.println("");
    la.push(5);
    la.push(6);
    la.push(3);
    la.push(8);
    for(int t : la.queue) {
      System.out.print(t + " ");
    }
    System.out.println("");
    la.push(7);
    for(int t : la.queue) {
      System.out.print(t + " ");
    }
    System.out.println("");
    la.push(9);
    for(int t : la.queue) {
      System.out.print(t + " ");
    }
    System.out.println("");
    la.push(10);
    la.push(11);
    for(int t : la.queue) {
      System.out.print(t + " ");
    }
    System.out.println("");
  }

  public static void PrintTopK(int[] numArray, int k) throws NullPointerException {
    assert (k > 0) : "Invalid window size. Require k > 0.";
  }
}

class Dequeue {
  public int[] queue;
  public int size;
  public boolean isFull;
  public int headIndex; // Points to first element
  public int tailIndex; // Points to last element

  Dequeue(int size) {
    assert size > 0: "Invalid queue size. Require size > 0.";
    this.queue = new int[size];
    this.size = size;
    this.headIndex = 0;
    this.tailIndex = -1;
    this.isFull = false;
  }

  public void push(int val) {
    if (this.isFull) {
      this.headIndex = (++this.headIndex) % size;
    }
    this.tailIndex = (++this.tailIndex) % size;
    this.queue[tailIndex] = val;
    if ((this.tailIndex+1) % size == this.headIndex) {
      this.isFull = true;
    } 
  }

  public void pushFlush(int val) {
    if (val >= this.queue[this.headIndex]) {
      this.headIndex = 0;
      this.tailIndex = 0;
      this.queue[this.headIndex] = val;
    } else {
      this.push(val);
    }
  }
}
