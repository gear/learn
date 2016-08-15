/*
 * Ex1-3 cracking
 */

public class StrURLify {
    public static void urlify(char[] rawStr, int trueLength) {
        if (rawStr == null) {
            System.out.println("WARNING. Null string input.");
            return;
        }
        if (rawStr.length == 0) {
            System.out.println("WARNING. Empty string input.");
            return;
        }
        StringBuilder cache = new StringBuilder();
        for (int i = 0; i < trueLength; ++i) {
            char c = rawStr[i];
            if (c == ' ')
                cache.append("%20");
            else {
                cache.append(c);
            }
        }
        cache.getChars(0, cache.length(), rawStr, 0);
    }
    public static void urlify_inline(char[] rawStr, int trueLength) {
        if (rawStr == null) {
            System.out.println("WARNING. Null string input.");
            return;
        }
        if (rawStr.length == 0) {
            System.out.println("WARNING. Empty string input.");
            return;
        }
        int shiftCount = rawStr.length - trueLength;
        for (int i = trueLength-1; i >= 0; --i) {
            if (shiftCount <= 0)
                return;
            if (rawStr[i] == ' ') {
                shiftCount -= 2;
                rawStr[i] = '%';
                rawStr[i+1] = '2';
                rawStr[i+2] = '0';
            } else {
                rawStr[i+shiftCount] = rawStr[i];
            }
        }
    }
    public static void main(String[] args) {
        if (args.length != 2) {
            System.out.println("ERROR. Invalid input.");
            return;
        }
        char[] achar = args[0].toCharArray();
        int trueLength = Integer.parseInt(args[1]);
        StrURLify.urlify_inline(achar, trueLength);
        System.out.println(achar);
    }
}
