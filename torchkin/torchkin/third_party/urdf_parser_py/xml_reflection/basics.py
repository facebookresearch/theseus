import string
import yaml
import collections.abc
from lxml import etree

def xml_string(rootXml, addHeader=True):
    # Meh
    xmlString = etree.tostring(rootXml, pretty_print=True, encoding='unicode')
    if addHeader:
        xmlString = '<?xml version="1.0"?>\n' + xmlString
    return xmlString


def dict_sub(obj, keys):
    return dict((key, obj[key]) for key in keys)


def node_add(doc, sub):
    if sub is None:
        return None
    if type(sub) == str:
        return etree.SubElement(doc, sub)
    elif isinstance(sub, etree._Element):
        doc.append(sub)  # This screws up the rest of the tree for prettyprint
        return sub
    else:
        raise Exception('Invalid sub value')


def pfloat(x):
    return str(x).rstrip('.')


def xml_children(node):
    children = node.getchildren()

    def predicate(node):
        return not isinstance(node, etree._Comment)
    return list(filter(predicate, children))


def isstring(obj):
    try:
        return isinstance(obj, basestring)
    except NameError:
        return isinstance(obj, str)


def to_yaml(obj):
    """ Simplify yaml representation for pretty printing """
    # Is there a better way to do this by adding a representation with
    # yaml.Dumper?
    # Ordered dict: http://pyyaml.org/ticket/29#comment:11
    if obj is None or isstring(obj):
        out = str(obj)
    elif type(obj) in [int, float, bool]:
        return obj
    elif hasattr(obj, 'to_yaml'):
        out = obj.to_yaml()
    elif isinstance(obj, etree._Element):
        out = etree.tostring(obj, pretty_print=True)
    elif type(obj) == dict:
        out = {}
        for (var, value) in obj.items():
            out[str(var)] = to_yaml(value)
    elif hasattr(obj, 'tolist'):
        # For numpy objects
        out = to_yaml(obj.tolist())
    elif isinstance(obj, collections.abc.Iterable):
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