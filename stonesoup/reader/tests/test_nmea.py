import datetime

import numpy as np
from io import StringIO

from stonesoup.reader.nmea import NMEASources, NMEAReader

ENGLISH_CHANNEL_BBOX = []

START_TIME = datetime.datetime(2019, 1, 6, hour=0, minute=0)

END_TIME = datetime.datetime(2019, 1, 6, hour=23, minute=59)

# The following string is an example containing NMEA sentences.
#  First two sentences are message type 1.
#  Third sentence is message type 18.
#  Next five sentences are message type 1.
#  Ninth sentence is message type 24.
#  Tenth sentence is a multi-line/multi-fragment (includes next line)
#  message of type 5.
#  Last sentence is message type 1 but occurs outside the time span.
#  In total there are 7 valid detections.
MSSIS_LOG = """
!ABVDM,1,1,5,A,13u=nOP0000jHt6Qdc9QS@qj0<0h,0*21,1546732800,1546732800
!SAVDM,1,1,5,B,14`UvE003<rC`et>qKERJB7f8<1J,0*73,1546732800,1546732860
!SAVDM,1,1,0,A,B52Ms6@006TFF23nT;WQ3wtUkP06,0*3D,1546732800,1546732920
!SAVDM,1,1,7,A,15NKj0gP@1IR:DNA9wvrNW;l08Qi,0*48,1546732800,1546733420
!ABVDM,1,1,7,B,13uKMPg001PnauTQ1<LeeWgj00SN,0*47,1546732800,1546737020
!SAVDM,1,1,7,B,15N0>1gP0pqW@NVAAOTq>gwl26sd,0*4F,1546732800,1546737040
!AIVDM,1,1,,A,13oBmv0000OplURRMLV8SF5j00S0,0*52,1546732800,1546737080
!SAVDM,1,1,0,B,15NGHBPP00I>TW@@v3G7FOwj0@Qm,0*11,1546732800,1546737180
!SAVDM,1,1,5,A,H52OqgPlu8pTpN1=@5800000000,2*37,1546732800,1546737480
!ABVDM,2,1,9,B,537`0482?HMP?AT?8004ppF1<T9Dl00000000017AP<1G4C10AliAkm00000,0*1E,1546737980
!ABVDM,2,2,9,B,00000000000,2*25,1546732800,1546737980
!AIVDM,1,1,,B,177KQJ5000G?tO`K>RA1wUbN0TKH,0*5C,1546696800,1546696860
"""


EA_LOG = """
\s:EXACTEARTH-A,q:u,c:1546732791*5F\!ABVDM,1,1,5,A,13u=nOP0000jHt6Qdc9QS@qj0<0h,0*21,1546732800,1546732800.70
\g:2-2-0335*58\!SAVDM,1,1,5,B,14`UvE003<rC`et>qKERJB7f8<1J,0*73,1546732800,1546732860.70
\g:2-2-0335*58\!SAVDM,1,1,0,A,B52Ms6@006TFF23nT;WQ3wtUkP06,0*3D,1546732800,1546732920.70
\g:2-2-0335*58\!SAVDM,1,1,7,A,15NKj0gP@1IR:DNA9wvrNW;l08Qi,0*48,1546732800,1546733420.70
\s:RCM-A,q:u,c:1546732793*5D\!ABVDM,1,1,7,B,13uKMPg001PnauTQ1<LeeWgj00SN,0*47,1546732800,1546737020.59
\s:RCM-C,q:u,c:1546732793*5D\!SAVDM,1,1,7,B,15N0>1gP0pqW@NVAAOTq>gwl26sd,0*4F,1546732800,1546737040.70
\s:RCM-B,q:u,c:1546732793*5D\!AIVDM,1,1,,A,13oBmv0000OplURRMLV8SF5j00S0,0*52,1546732800,1546737080.70
\s:EXACTEARTH-C,q:u,c:1546732791*5F\!SAVDM,1,1,0,B,15NGHBPP00I>TW@@v3G7FOwj0@Qm,0*11,1546732800,1546737180.70
\g:2-2-0335*58\!SAVDM,1,1,5,A,H52OqgPlu8pTpN1=@5800000000,2*37,1546732800,1546737480.70
\s:RCM-A,q:u,c:1546732793*5D\!ABVDM,2,1,9,B,537`0482?HMP?AT?8004ppF1<T9Dl00000000017AP<1G4C10AliAkm00000,0*1E,1546737980.70
\s:EXACTEARTH-B,q:u,c:1546732791*5F\!ABVDM,2,2,9,B,00000000000,2*25,1546732800,1546737980.69
\s:EXACTEARTH-C,q:u,c:1546732791*5F\!AIVDM,1,1,,B,177KQJ5000G?tO`K>RA1wUbN0TKH,0*5C,1546696800,1546696860.29
"""

# Decoded test data as csv.
TEST_DATA = np.loadtxt(StringIO("""
11.00779167,58.89158333,0.,2019,1,6,0,0,0,265516670
-79.59648333,26.03491667,0.,2019,1,6,0,1,0,311000660
-90.404455,29.98271167,0.,2019,1,6,0,10,20,367456770
11.93971,57.70429667,0.,2019,1,6,1,10,20,265739650
-89.29116833,30.18733833,0.,2019,1,6,1,10,40,265739650
-1.56813167,60.2234,0.,2019,1,6,1,11,20,259307000
-94.68377333,29.65647333,0.,2019,1,6,1,13,0,367384650
"""), delimiter=",", dtype=float)


def _test_detections(detections):
    assert len(detections) == 7
    for n, detection in enumerate(detections):
        assert np.equal(round(TEST_DATA[n, 0], 5),
                        round(detection.state_vector[0][0], 5))
        assert np.equal(round(TEST_DATA[n, 1], 5),
                        round(detection.state_vector[1][0], 5))
        assert np.equal(round(TEST_DATA[n, 2], 5),
                        round(detection.state_vector[2][0], 5))
        assert int(TEST_DATA[n, 3]) == detection.timestamp.year
        assert int(TEST_DATA[n, 4]) == detection.timestamp.month
        assert int(TEST_DATA[n, 5]) == detection.timestamp.day
        assert int(TEST_DATA[n, 6]) == detection.timestamp.hour
        assert int(TEST_DATA[n, 7]) == detection.timestamp.minute
        assert int(TEST_DATA[n, 8]) == detection.timestamp.second
        assert int(TEST_DATA[n, 9]) == detection.metadata['mmsi']


def test_nmea_mssis(tmpdir):

    # create test mssis file
    mssis_filename = tmpdir.join("mssis_nmea.log")
    mssis_file = mssis_filename.open('w')
    mssis_file.write(MSSIS_LOG)
    mssis_file.close()

    # Read the MSSIS file with a "NMEAReader()" to extract data.
    mssis_reader = NMEAReader(start_time=START_TIME,
                              end_time=END_TIME,
                              path=mssis_filename,
                              src=NMEASources.MSSIS)
    detections = [detection
                  for _, detections in mssis_reader.detections_gen()
                  for detection in detections]

    # verify that all of the AIS records from the NMEA file were read correctly
    _test_detections(detections)


def test_nmea_ealog(tmpdir):

    # create test mssis file
    ealog_filename = tmpdir.join("ealog_nmea.log")
    ealog_file = ealog_filename.open('w')
    ealog_file.write(EA_LOG)
    ealog_file.close()

    # Read the Exact Earth file with a "NMEAReader()" to extract data.
    mssis_reader = NMEAReader(start_time=START_TIME,
                              end_time=END_TIME,
                              path=ealog_filename,
                              src=NMEASources.ExactEarth)
    detections = [detection
                  for _, detections in mssis_reader.detections_gen()
                  for detection in detections]

    # verify that all of the AIS records from the NMEA file were read correctly
    _test_detections(detections)
