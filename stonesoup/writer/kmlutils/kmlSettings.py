
KML_SCHEMA_FILE = "ogckml22.xsd"

GOOGLE_KML_SCHEMA_FILE = "kml22gx.xsd"


NS_MAP = {
        None: 'http://www.opengis.net/kml/2.2',
        'gx': 'http://www.google.com/kml/ext/2.2',
}


# Style directory  name
STYLE_DIR = "styles"

# Document path in KML file.
DOC_PATH = "/x:kml/x:Document"

# Document Name
DOC_NAME = "Stone Soup Output"



# Folder name for AIS tracks
TRACKS_FOLDER_NAME = 'All Stone Soup Tracks'


# KML track style nodes as string
STONE_SOUP_TRACK_STYLE_NODES_AS_STR = '''
	<SS_KML_STYLE_FORMAT>
	<Style id="{KML_STYLE_ID}_n">
      <IconStyle>
        <color>{KML_STYLE_COLOR}</color>
        <scale>0.5</scale>
        <Icon>
          <href>http://earth.google.com/images/kml-icons/track-directional/track-none.png</href>
        </Icon>
      </IconStyle>
    </Style>
    <Style id="{KML_STYLE_ID}_h">
      <IconStyle>
        <color>{KML_STYLE_COLOR}</color>
        <scale>1.2</scale>
        <Icon>
          <href>http://earth.google.com/images/kml-icons/track-directional/track-none.png</href>
        </Icon>
      </IconStyle>
    </Style>
    <StyleMap id="{KML_STYLE_ID}">
      <Pair>
        <key>normal</key>
        <styleUrl>#{KML_STYLE_ID}_n</styleUrl>
      </Pair>
      <Pair>
        <key>highlight</key>
        <styleUrl>#{KML_STYLE_ID}_h</styleUrl>
      </Pair>
    </StyleMap>
    <Style id="{KML_STYLE_ID}_L">
      <LineStyle>
        <color>{KML_STYLE_COLOR}</color>
        <width>6</width>
      </LineStyle>
    </Style>
	</SS_KML_STYLE_FORMAT>
'''

# KML detection style nodes as string
STONE_SOUP_DETECTION_STYLE_NODES_AS_STR = '''
	<SS_KML_STYLE_FORMAT>
	<Style id="{KML_STYLE_ID}_n">
      <IconStyle>
        <color>{KML_STYLE_COLOR}</color>
        <scale>0.5</scale>
        <Icon>
          <href>http://earth.google.com/images/kml-icons/track-directional/track-none.png</href>
        </Icon>
      </IconStyle>
    </Style>
    <Style id="{KML_STYLE_ID}_h">
      <IconStyle>
        <color>{KML_STYLE_COLOR}</color>
        <scale>1.2</scale>
        <Icon>
          <href>http://earth.google.com/images/kml-icons/track-directional/track-none.png</href>
        </Icon>
      </IconStyle>
    </Style>
    <StyleMap id="{KML_STYLE_ID}">
      <Pair>
        <key>normal</key>
        <styleUrl>#{KML_STYLE_ID}_n</styleUrl>
      </Pair>
      <Pair>
        <key>highlight</key>
        <styleUrl>#{KML_STYLE_ID}_h</styleUrl>
      </Pair>
    </StyleMap>
	</SS_KML_STYLE_FORMAT>
'''


# KML track style id prefix 
KML_TRACK_STYLE_ID_PREFIX = "SS_Style_Track"

# KML detection style id prefix 
KML_DETECTION_STYLE_ID_PREFIX = "SS_Style_Detection"

# KML style id place holder
KML_STYLE_ID_PLACEHOLDER = "{KML_STYLE_ID}"

# KML color place holder
KML_COLOR_PLACEHOLDER = "{KML_STYLE_COLOR}"