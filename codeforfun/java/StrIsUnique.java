/*Check if string has all unique characters.
 * Created: 2016-07-08
 * Author: Hoang NT
 * v0.0: StrIsUnique.java created
 * v0.1: HashSet version
 */

import java.util.HashSet;

public class StrIsUnique {
	boolean isUnique (String str_in) {
		/*Naive approach O(n^2)
		 * Iterate through the string. 
		 * Each iteration check the current
		 * character to see duplication 
		 * until the end of the string.
		 */
		// Check for empty string
		if (str_in.length() <= 0) {
			System.out.println("Empty or corrupted string input.");
			System.exit(1);
		}
		// Naive iteration 
		for (int i = 0; i < str_in.length()-1; ++i) {
			char curr = str_in.charAt(i);
			for (int j = i + 1; j < str_in.length(); ++j) {
				if (str_in.charAt(j) == curr)
					return false;
			}
		}
		return true;
	}

	boolean isUnique_Inline (String str_in) {
		/*No extra memory used for string checking. O(n)
		 * Iterate through the string and keep
		 * track of the "unique list" by swapping
		 * characters in the string. This type of
		 * implementation needs the input string
		 * to be an char array or StringBuffer.
		 */

		return true;
	}

	boolean isUnique_ASCII (String str_in) {
		/*Fixed memory requirement for the algorithm. O(n)
		 * This algorithm constrains the input string
		 * to be ASCII character only.
		 */
		char[] lookup_table = new char[256];
		for(char c : str_in.toCharArray()) {
			if (lookup_table[c] == 1)
				return false;
			else
				lookup_table[c] = 1;
		}
		return true;
	}

	boolean isUnique_HashSet (String str_in) {
        /*HashMap implementation for the algorithm. O(n)
         * This approach can get any symbol, not restricted
         * to ASCII at the cost of memory for the HashSet.
         */
        HashSet<Character> charset = new HashSet<Character>();
        for (char c : str_in.toCharArray()) {
            if ( charset.contains(c))
                return false;
            else
                charset.add(c);
        }
		return true;
	}

	public static void main(String[] args) {
		StrIsUnique test_obj = new StrIsUnique();
		System.out.println(test_obj.isUnique_HashSet(args[0]));
	}
}
