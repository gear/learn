/*
 * Ex2 - Chapter 1 - cracking
 */
import java.util.HashMap;
public class StrCheckPerm {
    static public boolean isPerm(String a, String b) {
        if (a == null || b == null) {
            System.out.println("Invalid input(s).");
            return false;
        }
        if (a.length() != b.length()) 
            return false;
        if (a.length() == 0)
            return true;
        HashMap<Character, Integer> cache = new HashMap<Character, Integer>();
        for (char c : a.toCharArray()) {
            int val = 1;
            if (cache.containsKey(c)) {
                val = cache.get(c);
                cache.put(c,++val);
            } else {
                cache.put(c, val);
            }
        }
        for (char c : b.toCharArray()) {
            int val = 0;
            try {
                val = cache.get(c);
            } catch(NullPointerException e) {
                return false;
            }
            cache.put(c,--val);
        }
        for (Integer i : cache.values()) 
            if (i != 0)
                return false;
        return true;
    }

    public static void main(String[] args) {
        if (args.length != 2) {
            System.out.println("Invalid command.");
            return;
        }
        String a = args[0];
        String b = args[1];
        System.out.println(StrCheckPerm.isPerm(a,b));
    }
}
