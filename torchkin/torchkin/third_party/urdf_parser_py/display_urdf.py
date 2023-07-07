import sys
import argparse

from torchkin.third_party.urdf_parser_py.urdf import URDF


def main():
    parser = argparse.ArgumentParser(usage='Load an URDF file')
    parser.add_argument('file', type=argparse.FileType('r'),
                        help='File to load. Use - for stdin')
    parser.add_argument('-o', '--output', type=argparse.FileType('w'),
                        default=None, help='Dump file to XML')
    args = parser.parse_args()

    robot = URDF.from_xml_string(args.file.read())

    print(robot)

    if args.output is not None:
        args.output.write(robot.to_xml_string())