import collections
import string
from xml.dom import minidom
from xml.etree import ElementTree as ET

import yaml


def xml_string(rootXml, addHeader=True):
    # From: https://stackoverflow.com/a/1206856/170413
    # TODO(eacousineau): This does not preserve attribute order. Fix it.
    dom = minidom.parseString(ET.tostring(rootXml))
    xml_string = ""
    lines = dom.toprettyxml(indent="  ").split("\n")
    if lines and lines[0].startswith("<?xml") and not addHeader:
        del lines[0]
    # N.B. Minidom injects some pure-whitespace lines. Remove them.
    return "\n".join(filter(lambda line: line.strip(), lines))


def dict_sub(obj, keys):
    return dict((key, obj[key]) for key in keys)


def node_add(doc, sub):
    if sub is None:
        return None
    if type(sub) == str:
        return ET.SubElement(doc, sub)
    elif isinstance(sub, ET.Element):
        doc.append(sub)  # This screws up the rest of the tree for prettyprint
        return sub
    else:
        raise Exception("Invalid sub value")


def pfloat(x):
    return str(x).rstrip(".")


def xml_children(node):
    return list(node)


def isstring(obj):
    try:
        return isinstance(obj, basestring)
    except NameError:
        return isinstance(obj, str)


def to_yaml(obj):
    """Simplify yaml representation for pretty printing"""
    # Is there a better way to do this by adding a representation with
    # yaml.Dumper?
    # Ordered dict: http://pyyaml.org/ticket/29#comment:11
    if obj is None or isstring(obj):
        out = str(obj)
    elif type(obj) in [int, float, bool]:
        return obj
    elif hasattr(obj, "to_yaml"):
        out = obj.to_yaml()
    elif isinstance(obj, type(ET.Element)):
        out = xml_string(obj, addHeader=False)
    elif type(obj) == dict:
        out = {}
        for var, value in obj.items():
            out[str(var)] = to_yaml(value)
    elif hasattr(obj, "tolist"):
        # For numpy objects
        out = to_yaml(obj.tolist())
    elif isinstance(obj, collections.Iterable):
        out = [to_yaml(item) for item in obj]
    else:
        out = str(obj)
    return out


class SelectiveReflection(object):
    def get_refl_vars(self):
        return list(vars(self).keys())


class YamlReflection(SelectiveReflection):
    def to_yaml(self):
        raw = dict((var, getattr(self, var)) for var in self.get_refl_vars())
        return to_yaml(raw)

    def __str__(self):
        # Good idea? Will it remove other important things?
        return yaml.dump(self.to_yaml()).rstrip()
