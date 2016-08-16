/*
 * Merge Overlapping Intervals - GeeksforGeeks
 */
import java.util.HashSet;
public class IntervalMerge {
    public static void main(String[] args) {
        HashSet<Interval<Integer>> iset = new HashSet<Interval<Integer>>();
        iset.add(new Interval<Integer>(1,5));
        iset.add(new Interval<Integer>(2,8));
        iset.add(new Interval<Integer>(8,12));
        iset.add(new Interval<Integer>(7,9));
        iset.add(new Interval<Integer>(3,4));
        iset.add(new Interval<Integer>(0,2));
        iset.add(new Interval<Integer>(9,13));
        iset.add(new Interval<Integer>(21,23));
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
    public Interval<T> left;
    public Interval<T> right;
    public Interval<T> self;
    public IntervalNode(T x, T y) {
        this.left = null;
        this.right = null;
        this.self = new Interval<T>(x, y);
    }
    public boolean add(Interval<T> elem) {
        /* Return false if a branch is added
         */
        if (elem == null)
            return true;
        if (!this.self.merge(elem)) {
            if (this.self.x.compareTo(elem.x) < 0) 
                this.left = elem;
            else
                this.right = elem;
            return false;
        }
        return true;
    }
}
