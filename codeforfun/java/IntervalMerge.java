/*
 * Merge Overlapping Intervals - GeeksforGeeks
 */
import java.util.HashSet;
public class IntervalMerge {
    public static void main(String[] args) {
        HashSet<Interval<Integer>> iset = new HashSet<Interval<Integer>>();
        iset.add(new Interval<Integer>(1,20));
        iset.add(new Interval<Integer>(4,6));
        iset.add(new Interval<Integer>(29,41));
        IntervalNode<Integer> root = null;
        for(Interval<Integer> e : iset) {
            if (root == null)
                root = new IntervalNode<Integer>(e);
            else
                if (root.add(e))
                    root.recursiveMerge();
        }
        root.print();
    }
}

class Interval<T extends Comparable<T>> {
    public T x;
    public T y;
    public Interval(T x, T y) {
        this.x = x;
        this.y = y;
    }
    public boolean merge(Interval<T> other) {
        /* Return false if the other interval
         * is exclusive
         */
        if (other == null)
            return false;
        if (this.x.compareTo(other.x) > 0) {
            if (this.x.compareTo(other.y) > 0)
                return false;
            else {
                if (this.y.compareTo(other.y) > 0) {
                    this.x = other.x;
                    return true;
                } else {
                    this.x = other.x;
                    this.y = other.y;
                    return true;
                }
            }
        } else {
            if (this.y.compareTo(other.x) < 0)
                return false;
            else {
                if (this.y.compareTo(other.y) < 0) {
                    this.y = other.y;
                    return true;
                } else 
                    return true;
            }
        }
    }
    public void print() {
        System.out.println("(" + this.x + "," + this.y + ")");
    }
}

class IntervalNode<T extends Comparable<T>> {
    public IntervalNode<T> left;
    public IntervalNode<T> right;
    public Interval<T> self;
    public IntervalNode(T x, T y) {
        this.left = null;
        this.right = null;
        this.self = new Interval<T>(x, y);
    }
    public IntervalNode(Interval<T> init) {
        this.left = null;
        this.right = null;
        this.self = init;
    }
    public boolean add(Interval<T> elem) {
        /* Return false if a branch is added
         */
        if (elem == null)
            return true;
        if (!this.self.merge(elem)) {
            if (this.self.x.compareTo(elem.x) < 0) 
                if (this.left == null) 
                    this.left = new IntervalNode<T>(elem);
                else
                    this.left.add(elem);
            else
                if (this.right == null)
                    this.right = new IntervalNode<T>(elem);
                else
                    this.right.add(elem);
            return false;
        }
        return true;
    }
    public Interval<T> recursiveMerge() {
        if (this.left != null)
            this.self.merge(left.recursiveMerge());
        if (this.right != null)
            this.self.merge(right.recursiveMerge());
        return this.self;
    }
    public void print() {
        if (this.left != null)
            this.left.print();
        if (this.right != null)
            this.right.print();
        this.self.print();
    }
}
