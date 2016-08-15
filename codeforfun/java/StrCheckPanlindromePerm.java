/*
 * Ex1-4 - cracking
 */
public class StrCheckPanlindromePerm {
    public static boolean isPanlindromePerm(char[] str) {
        int[] asciiCount = new int[256];
        for (char c : str) {
            if (c != ' ') {
                asciiCount[c]++;
            }
        }
        boolean odd_found = false;
        for (int i : asciiCount) {
            if (i % 2 != 0) 
                if (odd_found)
                    return false;
                else
                    odd_found = true;
        }
        return true;
    }
    public static void main(String[] args) {
        if (args.length != 1) {
            System.out.println("ERROR. Invalid terminal input.");
            return;
        }
        System.out.println(StrCheckPanlindromePerm.isPanlindromePerm(args[0].toCharArray()));
    }
}
