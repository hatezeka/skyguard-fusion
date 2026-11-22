import argparse, json
from .criteria import from_file

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--evidence',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    print(json.dumps(from_file(a.evidence,a.out),indent=2))

if __name__=='__main__': main()
