public class OneAway {
  public static boolean IsOne(char[] org, char[] edited) {
    int lengthDiff = org.length - edited.length;
    if ((lengthDiff > 1) || (lengthDiff < -1))
      return false;
    int i = 0;
    int j = 0;
    while ((i < org.length-1) && (j < edited.length-1)) {
      if (org[i] != edited[j]) {
        break;
      } else {
        ++i; ++j;
      }
    }
    int ii = org.length-1;
    int jj = edited.length-1;
    while ((ii > i) && (jj > j)) {
      if (org[ii] != edited[jj]) {
        break;
      } else {
        --ii; --jj;
      }
    }
    if (i == ii) {
      if (j == jj) {
        return true; // Zero or replace
      } else {
        return ((org[i] == edited[j]) || (org[i] == edited[jj]));
      }
    } else {
      if (j == jj) {
        return ((edited[j] == org[i]) || (edited[j] == org[ii]));
      } else {
        return false; // More than 1
      }
    }
  }

  public static void main(String[] args) {
    if (args.length != 2)
      System.out.println("Wrong input format.");
    String org = args[0];
    String edited = args[1];
    System.out.println(OneAway.IsOne(org.toCharArray(), edited.toCharArray()));
  }
}

