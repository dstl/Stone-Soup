import functools


class InvalidMessage(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class NMEADecoder(object):
    ais_six_bit_text =\
        "@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_ !\"#$%&\\()*+,-./0123456789:;<=>?"
    payload = ""
    decode_dictionary = {}

    def __init__(self, payload=""):
        if (payload):
            self.setPayload(payload)
        self.decode_dictionary = {'int': self.decodeInt,
                                  'uint': self.decodeUint,
                                  'hexbits': self.decodeHexbits,
                                  'float': self.decodeFloat}

    def setPayload(self, payload):
        self.payload = payload

    def joinFields(self, source, masks, shifts):
        """
        Split the ASCII text in hexbit fields with appropriate masks
        and shifts.
        """
        six_bit = [(ord(x) - 48 - int(ord(x) > 88) * 8) & y
                   for x, y in zip(source, masks)]
        try:
            return
            functools.reduce(lambda x, y: x | y,
                             [x << y
                              for x, y in zip(six_bit, shifts) if y >= 0] +
                             [x >> abs(y)
                              for x, y in zip(six_bit, shifts) if y < 0])
        except TypeError:
            raise InvalidMessage(self.payload)

    def decodeHexbits(self, source, element):
        """
        Return the value as a 6-bit ASCII charater accordin to
        six-bit map table.
        """
        return self.ais_six_bit_text[self.decodeUint(source, element)]

    def decodeUint(self, source, element):
        """
        Return an unsigned integer
        """
        return self.joinFields(source, element['masks'], element['bitShifts'])

    def decodeInt(self, source, element):
        """
        Return a signed integer
        """
        raw_int = self.decodeUint(source, element)
        sign_bit = 0
        if('signBit' in element):
            sign_bit = int(element['signBit'], 16)
        return raw_int - 2 * (raw_int & sign_bit)

    def decodeFloat(self, source, element):
        """
        Return a floating point value
        """
        scale = 1.0
        if ('scale' in element):
            scale = float(element['scale'])
        return float(self.decodeInt(source, element)) / scale

    def decodeField(self, field):
        """
        Decode a field
        """
        return [self.decodeElement(element) for element in field.elements]

    def decodeElement(self, element):
        """
        Decode an element of the field.
        """
        if (len(element) < 3):
            return []

        return self.decode_dictionary[element['type']](
            self.payload[element['index'][0]:element['index'][-1]],
            element)
