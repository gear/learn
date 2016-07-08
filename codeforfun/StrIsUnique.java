/*Check if string has all unique characters.
 * Created: 2016-07-08
 * Author: Hoang NT
 * v0.0: StrIsUnique.java created
 */

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
		/*No extra memory used for string checking.
		 * Iterate through the string and keep
		 * track of the "unique list" by swapping
		 * characters in the string. This type of
		 * implementation needs the input string
		 * to be an char array or StringBuffer
		return true;
	}

	boolean isUnique_ASCII (String str_in) {
		return true;
	}

	boolean isUnique_HashMap (String str_in) {
		return true;
	}

	public static void main(String[] args) {
		StrIsUnique test_obj = new StrIsUnique();
		System.out.println(test_obj.isUnique(args[0]));
	}
}