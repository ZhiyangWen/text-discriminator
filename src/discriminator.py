import argparse #for implementing basic command line application
import sys #for reading stdin
import numpy
from features.stylometry import extract_stylometry
from features.structural import extract_structural
from features.sequence import extract_sequence

print("is running")

def main():
    parser = argparse.ArgumentParser(description="Extract three stream feature")
    parser.add_argument("input", help= "read the text or when '-' read the input from user")
    args = parser.parse_args()
    if args.input == "-":
        text = sys.stdin.read()

    else:
        with open(args.input,"r", encoding="utf-8") as f:
            text = f.read()
    styl = extract_stylometry(text)
    struc = extract_structural(text)
    seq = extract_sequence(text)
    print("stylometry:", styl.tolist())
    print("structural:", struc.tolist())
    print("sequence:", seq.tolist())
    whole = numpy.concatenate([
        styl,struc,seq
    ])
    print("whole:", whole.tolist())

if __name__ == "__main__":
    main()