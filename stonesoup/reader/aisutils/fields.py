import re


class AISField(object):
	name = ""
	elements = []

	def __init__(self, xml_desc=[]):
		if (xml_desc is not None):
			self.setFromXML(xml_desc)

	def setFromXML(self, xml_desc):
		self.name = xml_desc.attrib['name']
		self.elements = [self.decodeXML(el) for el in xml_desc]

	def decodeXML(self, element):
		if (len(element) < 3):
			return []
		idx = [int(x) for x in re.split('[^0-9\-]', element[0].text)]
		masks = [int(x, 16) for x in re.split('[^0-9a-zA-Z]',element[1].text)]
		bit_shifts = [int(x) for x in re.split('[^0-9\-]', element[2].text)]
		return dict({'index': idx, 'masks': masks, 'bitShifts': bit_shifts},
				**element.attrib)