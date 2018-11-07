import matplotlib
from matplotlib import cm

from lxml import etree


import kmlSettings as settings

def appendSSStyles(doc_node, color_count = 5):
	norm = matplotlib.colors.Normalize(vmin=1, vmax=colour_count)
	for k in range(1, colour_count):
		styleIDPre = format("{}_{}".format(settings.KML_STYLE_ID_PREFIX, k))
		styleColorRGBA = cm.jet(norm(k))
		styleColorHEX = matplotlib.colors.rgb2hex(styleColorRGBA)
		# Matplotlib HEX -> #RGBA, KML HEX -> #ABGR
		styleColorKMLHEX = "#ff{}".format(styleColorHEX[6:0:-1])
		# Generate Style Nodes
		ssKMLStylesStr = settings.STONE_SOUP_STYLE_NODES_AS_STR
		ssKMLStylesStr = ssKMLStylesStr.replace(settings.KML_STYLE_ID_PLACEHOLDER, styleIDPre)
		ssKMLStylesStr = ssKMLStylesStr.replace(settings.KML_COLOR_PLACEHOLDER, styleColorKMLHEX)
		ssStyleNodes = etree.fromstring(ssKMLStylesStr)
		styleNodes = ssStyleNodes.getchildren()
		[doc_node.append(sNode) for sNode in styleNodes]




def ssKMLDocFactory(num_tracks=5):
	kmlRoot = etree.Element('kml', nsmap=settings.NS_MAP)
	kmlDoc = etree.SubElement(kmlRoot, "Document")
	docName = etree.SubElement(kmlDoc, "name")
	docName.text = settings.DOC_NAME
	numStyles = appendSSStyles(kmlDoc, num_tracks)
	tksFolder = etree.SubElement(kmlDoc, "Folder")
	tksFolderName = etree.SubElement(tksFolder, "name")
	tksFolderName.text = "All Tracks"
	


