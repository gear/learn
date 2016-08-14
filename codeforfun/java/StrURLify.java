/*
 * Ex1-3 cracking
 */

public class StrURLify {
    public static void urlify(char[] rawStr) {
        if (rawStr == null) {
            System.out.println("WARNING. Null string input.");
            return;
        }
        if (rawStr.length == 0) {
            System.out.println("WARNING. Empty string input.");
            return;
        }
        StringBuilder cache = new StringBuilder();
        for (char c : rawStr) {
            if (c == ' ')
                cache.append("%20");
            else
                cache.append(c);
        }
        cache.getChars(0, cache.length()-1, rawStr, 0);
    }
    public static void urlify_inline(char[] rawStr) {
        if (rawStr == null) {
            System.out.println("WARNING. Null string input.");
            return;
        }
        if (rawStr.length == 0) {
            System.out.println("WARNING. Empty string input.");
            return;
        }
    }
    public static void main(String[] args) {
        if (args.length != 1) {
            System.out.println("ERROR. Invalid input.");
            return;
        }
        char[] achar = args[0].toCharArray();
        StrURLify.urlify(achar);
        System.out.println(achar);
    }

}
