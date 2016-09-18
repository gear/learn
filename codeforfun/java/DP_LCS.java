import java.util.*;
import java.io.*;

public class DP_LCS {

    public static void main(String[] args) {
        assert (args.length == 2) : "Invalid input.";
        String str1 = args[0];
        String str2 = args[1];
        System.out.println(lcs(str1,str2));
    }

    public static int mem(int i, int j, int[][] arr) {
        if (i < 0 || j < 0)
            return 0;
        else
            return arr[i][j];
    }

    public static int max(int a, int b) {
        return (a >= b) ? a : b;
    }

    public static int lcs(String a, String b) {
        int[][] arr = new int[a.length()][b.length()];
        for (int i = 0; i < a.length(); ++i) {
            for (int j = 0; j < b.length(); ++j) {
                if (a.charAt(i) == b.charAt(j)) {
                    arr[i][j] = mem(i-1,j-1,arr) + 1;
                } else {
                    arr[i][j] = max(mem(i,j-1,arr),mem(i-1,j,arr)); 
                }
            }
        }
        int i = arr[0].length;
        int oldVal = arr[a.length()-1][b.length()-1];
        char[] strLCS = new char[oldVal];
        while (i >= 0) {
          --i;
          if(mem(a.length()-1,i,arr) != oldVal) {
            oldVal = mem(a.length()-1,i,arr);
            System.out.print(b.charAt(i+1));
          }
        }
        System.out.println();
        return arr[a.length()-1][b.length()-1];
    }
}
