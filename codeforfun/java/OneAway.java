public class OneAway {
  public static int editDistance(String org, String edited) {
    int lengthDiff = org.length() - edited.length();
    if ((lengthDiff > 1) || (lengthDiff < -1))
      return -1;
    String shorter = org;
    String longer = edited;
    if (lengthDiff == 1)
      shorter = edited;
      longer = org;
    int j = 0;
    int i = 0;
    boolean diff = false;
    while (i < shorter.length()) {
      if (shorter.charAt(i) != longer.charAt(j)) {
        if (diff) 
          return -1;
        else {
          diff = true;
          ++j;
        }
      } else {
        ++i;
        ++j;
      }
    }
    if ((lengthDiff==0) && diff)
      return 0;
    else
      return 1;
  }
  public static void main(String[] args) {
    if (args.length != 2)
      System.out.println("Wrong input format.");
    String org = args[0];
    String edited = args[1];
    System.out.println(OneAway.editDistance(org, edited));
  }
}

