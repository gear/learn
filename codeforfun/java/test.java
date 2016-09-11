import java.io.*;
import java.util.*;
import java.text.*;
import java.math.*;
import java.util.regex.*;

public class test {
  public static void main(String[] args) {
    int i = 4;
    double d = 4.0;
    String s = "HackerRank ";

    Scanner scan = new Scanner(System.in);
    int ii;
    double dd;
    String ss;
    ii = scan.nextInt();
    dd = scan.nextDouble();
    scan.nextLine();
    ss = scan.nextLine();
    System.out.println(i + ii);
    System.out.println(d + dd);
    System.out.println(s + ss);
    scan.close();
  }
}
