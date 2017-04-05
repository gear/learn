"""Find html tags using regex"""
import re


def findtags(text):
    tags_pattern = r"""<\s*\w+((\s+\w+(\s*=\s*(".*?"|'.*?'))?)+\s*|\s*)>"""
    start_index = 0
    tags = []
    while start_index < len(text):
        tag = re.search(tags_pattern, text[start_index:])
        if not tag:
            return tags
        else:
            tags.append(text[start_index+tag.start():start_index+tag.end()])
            start_index += tag.end()
    return tags

texttest1 = """
My favorite website in the world is probably
<a href="www.udacity.com">Udacity</a>. If you want
that link to open in a <b>new tab</b> by default, you should
write <a href="www.udacity.com"target="_blank">Udacity</a>
instead!
"""

testtext2 = """
Okay, so you passed the first test case. <let's see> how you
handle this one. Did you know that 2 < 3 should return True?
So should 3 > 2. But 2 > 3 is always False.
"""

testtext3 = """
It's not common, but we can put a LOT of whitespace into
our HTML tags, For example, we can make something bold by
doing <           b            > this <       /b    >. Though I
don't know why you would ever want to.
"""

def test():
    assert findtags(texttest1) == ['<a href="www.udacity.com">',
                                  '<b>',
                                  '<a href="www.udacity.com"target="_blank">']
    assert findtags(testtext2) == []
    assert findtags(testtext3) == ['<           b            >']
    return 'tests pass'


def main():
    print(test())


if __name__ == '__main__':
    main()
