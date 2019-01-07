import re
import functools

from datetime import datetime, date, time


class NMEA(object):
    """
    This is the top-level parent/abstract class for NMEA parser classes.

    Whereas the raw AIS message format is (mostly) standardised, different
    providers/sources can prepend/append other fields to the data
    and organise their files in any manner as they see fit. For example,
    various providers of space-based AIS like to prepend multiple timestamps
    and other ancilliary information as the data passes through their system.
    Maritime Safety & Security Information System (MSSIS) provides data in time
    ascending or geo-sorted manner. Therefore, we need slightly different
    parsers depending on the source. That said, most AIS data can be parsed
    using either MSSIS (MSSISParser) or ExactEarth (EAParser) parser.

    Do not instantiate from this class. Instead, use one of the sub-classes.
    """

    valid_types = ['!AIVDM', '!ANVDM', '!ASVDM', '!ABVDM', '!BSVDM']
    reference_utc = datetime.combine(date(1970, 1, 1), time(0, 0, 0))
    count = 0
    max_number = -1
    time_span = [-1, -1]

    def __init__(self, filepath='',
                 reference_time=datetime.combine(
                     date(1970, 1, 1), time(0, 0, 0)),
                 max_number=-1, time_span=[-1, -1]):
        self.max_number = max_number
        self.reference_utc = reference_time
        self.ais_file = None
        self.setTimeSpan(time_span)
        self.open(filepath)

    def __iter__(self):
        return self

    def open(self, filepath):
        try:
            self.ais_file = open(filepath, 'r')
        except Exception as e:
            # Log?
            print("Error! Unable to open file at {}".format(filepath))
            print(type(e))
            raise

    def __next__(self):
        # Read a line from the file
        ln = self.ais_file.readline()
        # Check if a line was read
        if (not ln):
            raise StopIteration
        # Check if we have reached the maximum number of lines
        if (self.max_number > 0 and self.count > self.max_number):
            raise StopIteration

        # Parse the line
        target_line = self.splitLine(ln)

        # Check if time is within the time_span.
        if (self.time_span[-1] > 0 and
                int(target_line[-1]) > self.time_span[-1]):
            raise StopIteration
        # Update the count
        self.updateCount()

        return target_line

    def splitLine(self, line):
        raise NotImplementedError("Subclass should implement abstract method.")

    def close(self):
        self.ais_file.close()

    def checkSum(self, src):
        """
        Perform a checksum on a line of data.
        """
        m = re.search('[0-9A-F]\*[0-9A-F][0-9A-F]', src)
        if (m):
            chksum_bdy = src[1:m.start() + 1]
            return functools.reduce(lambda x, y: x ^ y,
                                    [ord(x) for x in chksum_bdy])
        return 0xFF

    def checkForMultiLine(self, fields):
        """
        Check if this is a multiline message. If yes then
        retrieve the next part of the line and append the
        fields[5] array element. Reduce the line count because
        multiline message counts as only one message

        Inputs: 1. fields: Array of fields parsed from a line.
        """
        if (int(fields[1]) > int(fields[2])):
            dummy = self.__next__()
            if (int(dummy[2]) - int(fields[2]) == 1):
                fields[5] = fields[5] + dummy[5]
                self.count -= 1
            else:
                return self.__next__()
            if (len(dummy) > len(fields)):
                fields.append(dummy[-1])
        return fields

    def setTimeSpan(self, time_span, advance_time=False):
        self.time_span = [-1, -1]
        if (time_span[0] > 0 and advance_time):
            self.advanceToTime(time_span[0])
        self.time_span = time_span

    def advanceToTime(self, t_advance):
        """
        For ordered log files only.
        Advance the file read point by t_advance.

        Inputs: 1. t_advance: Time to advance the file
                        read point by.
        """
        if (not self.ais_file):
            return False

        # Fet the time of the first line.
        self.ais_file.seek(0)
        first_line = self.__next__()
        self.count = 0

        max_msg_size = 82 * 10
        self.ais_file.seek(-max_msg_size, 2)
        last_line = None

        while True:
            try:
                last_line = self.__next__()
                self.count = 0
            except StopIteration:
                break
        if (last_line is None):
            # Time requested out of range.
            self.ais_file.seek(0)
        else:
            file_size = self.ais_file.tell()
            time_span = [int(first_line[-1]), int(last_line[-1])]

            if (t_advance < time_span[0] or t_advance > time_span[-1]):
                # Time requested out of range.
                self.ais_file.seek(0)
            else:
                fraction = (float(t_advance - time_span[0]) /
                            (time_span[-1] - time_span[0]))
                target_time = self.seekFraction(fraction, file_size)

                while target_time > t_advance:
                    fraction = fraction * 0.99
                    target_time = self.seekFraction(fraction, file_size)

                file_pos = self.ais_file.tell()
                while target_time < t_advance:
                    file_pos = self.ais_file.tell()
                    target_time = self.__next__RecordTime()
                self.ais_file.seek(file_pos)

    def seekFraction(self, fraction, filesize):
        """
        Advance the file read point to a point given by
        a fraction of the file size.
        """
        offset = int(float(filesize) * fraction)
        self.ais_file.seek(offset)
        return self.__next__RecordTime()

    def nextRecordTime(self):
        """
        Get the time of the next message.
        """
        try:
            target_line = self.__next__()
            self.count = 0
            return int(target_line[-1])
        except StopIteration:
            return

    def updateCount(self):
        """"
        Update counter.
        """
        self.count += 1


class MSSISParser(NMEA):
    """
    Parser for NMEA logs from MSSIS (https://mssis.volpe.dot.gov/Main/).
    These logs are usually, but not always, ordered in time.

    """

    def __init__(self, filepath='',
                 reference_time=datetime.combine(
                     date(1970, 1, 1), time(0, 0, 0)),
                 max_number=-1, time_span=[-1, -1]):
        super().__init__(filepath, reference_time, max_number, time_span)

    def splitLine(self, line):
        fields = line.split(',')

        # Check for the right kind of message.
        if fields[0] not in self.valid_types:
            return self.__next__()
        # Do a checksum on the data.
        # Go to next item upon failure.
        chksum = '*{:2X}'.format(self.checkSum(line))
        if chksum in "*FF":
            return self.__next__()
        if chksum not in fields[6]:
            return self.__next__()
        # Check if this is a multiline message.
        return self.checkForMultiLine(fields)


class EAParser(NMEA):
    """
    Parser for NMEA logs from ExactEarth or other space-based providers.
    These providers typically prepend the logs with some ancilliary information
    and therefore need to be parsed a bit differently from MSSIS.
    """

    def __init__(self, filepath='',
                 reference_time=datetime.combine(
                     date(1970, 1, 1), time(0, 0, 0)),
                 max_number=-1, time_span=[-1, -1]):
        super().__init__(filepath, reference_time, max_number, time_span)

    def splitLine(self, line):
        # Split the line into fields.
        # First get rid of the stuff at the front
        tmp_fields = line.split('\\')
        fields = tmp_fields[-1].split(',')

        # Check for the right kind of message.
        if fields[0] not in self.valid_types:
            return self.__next__()

        # See if we can get a time-string
        time_str = "0"
        if (len(tmp_fields) > 1):
            time_idx = tmp_fields[1].find('c:')
            if (time_idx):
                time_str = tmp_fields[1][(time_idx + 2):]
                time_str = time_str[0:time_str.find('*')]

        # Append the time string twice so we have the same format as MSSIS
        fields.append(time_str)
        fields.append(time_str)

        # Do a checksum on the data.
        # Go to next item upon failure.
        chksum = '*{:2X}'.format(self.checkSum(line))
        if chksum in "*FF":
            return self.__next__()
        if chksum not in fields[6]:
            return self.__next__()
        # Check if this is a multiline message.
        return self.checkForMultiLine(fields)


class MiscParser(EAParser):
    def __init__(self, filepath='',
                 reference_time=datetime.combine(
                     date(1970, 1, 1), time(0, 0, 0)),
                 max_number=-1, time_span=[-1, -1]):
        super().__init__(filepath, reference_time, max_number, time_span)
