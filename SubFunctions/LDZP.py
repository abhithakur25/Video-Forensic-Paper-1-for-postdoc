import numpy as np


# Local Directional ZigZag Pattern
class LocalDirectionalZigZagPattern(object):
    def __init__(self, image):
        self.image = image

    @staticmethod
    def ZigZag(I2, mapping, mode):
        """
        Returns the local ZigZag pattern histogram of an image depending on the mapping used.
        Possible values for mode are:
            - 'h' or 'hist'  to get a histogram of LZP codes
            - 'nh'           to get a normalized histogram

        Parameters:
        I2 : numpy.ndarray
            Input image.
        mapping : object
            The mapping table with a `num` attribute (number of bins) and `table` method for the lookup.
        mode : str
            Mode to return histogram or normalized histogram.

        Returns:
        numpy.ndarray
            Histogram or normalized histogram based on the mode.
        """

        # Get the dimensions of the image
        m, n = I2.shape

        # Initialize an empty result array
        result = np.zeros_like(I2)

        # Loop through the image (ignoring the borders)
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                J0 = I2[i, j]

                # Create a binary representation based on neighbors
                I3 = np.zeros((m, n), dtype=bool)
                I3[i - 1, j - 1] = I2[i - 1, j - 1] > J0
                I3[i - 1, j] = I2[i - 1, j] > J0
                I3[i - 1, j + 1] = I2[i - 1, j + 1] > J0
                I3[i, j + 1] = I2[i, j + 1] > J0
                I3[i + 1, j + 1] = I2[i + 1, j + 1] > J0
                I3[i + 1, j] = I2[i + 1, j] > J0
                I3[i + 1, j - 1] = I2[i + 1, j - 1] > J0
                I3[i, j - 1] = I2[i, j - 1] > J0

                # Calculate the result for rotational order 1
                result[i, j] = (
                        I3[i - 1, j - 1] * 2 ** 0 + I3[i - 1, j] * 2 ** 1 + I3[i, j - 1] * 2 ** 2 + I3[
                    i + 1, j - 1] * 2 ** 3 +
                        I3[i - 1, j + 1] * 2 ** 4 + I3[i, j + 1] * 2 ** 5 + I3[i + 1, j] * 2 ** 6 + I3[
                            i + 1, j + 1] * 2 ** 7
                )

        # Apply mapping if it is defined
        if isinstance(mapping, dict):  # Assuming mapping is a dictionary-like object
            bins = mapping['num']
            for i in range(result.shape[0]):
                for j in range(result.shape[1]):
                    result[i, j] = mapping['table'][result[i, j]]

        # Return histogram based on the mode
        if mode in ['hist', 'nh']:
            Pattern, _ = np.histogram(result.flatten(), bins=np.arange(0, bins))

            if mode == 'nh':
                Pattern = Pattern / np.sum(Pattern)

            return Pattern

        else:
            return result

    @staticmethod
    def bitget(number, positions):
        """
        Get the bits of a number at specific positions.

        Parameters:
        number : int
            The number from which to extract the bits.
        positions : list of int
            The positions of the bits to extract (1-based indexing).

        Returns:
        list of int
            A list of the bit values at the specified positions.
        """
        return [(number >> (pos - 1)) & 1 for pos in positions]

    def getmapping_u2(self, samples):
        """
        Uniform 2 mapping: Uniform patterns with up to 2 transitions.
        """

        table = np.zeros(2 ** samples, dtype=int)
        newMax = 0
        index = 0

        for i in range(2 ** samples):
            # Perform a left circular shift (rotate)
            j = np.bitwise_or(np.left_shift(i, 1), np.bitwise_and(i, 2 ** (samples - 1) - 1))  # Rotate left
            numt = np.sum(self.bitget(np.bitwise_xor(i, j), np.arange(1, samples + 1)))

            if numt <= 2:
                table[i] = index
                index += 1
            else:
                table[i] = newMax - 1

        return {'table': table, 'samples': samples, 'num': newMax}

    @staticmethod
    def getmapping_ri(samples):
        """
        Rotation Invariant mapping.
        """

        table = np.zeros(2 ** samples, dtype=int)
        tmpMap = np.full(2 ** samples, -1, dtype=int)
        newMax = 0

        for i in range(2 ** samples):
            rm = i
            r_bin = np.array(list(np.binary_repr(i, width=samples)), dtype=int)

            for j in range(1, samples):
                r_bin = np.roll(r_bin, -1)  # Rotate left
                r = int(''.join(r_bin.astype(str)), 2)

                if r < rm:
                    rm = r

            if tmpMap[rm] < 0:
                tmpMap[rm] = newMax
                newMax += 1

            table[i] = tmpMap[rm]

        return {'table': table, 'samples': samples, 'num': newMax}

    @staticmethod
    def getmapping_riu2(samples):
        """
        Uniform & Rotation Invariant mapping.
        """

        table = np.zeros(2 ** samples, dtype=int)
        newMax = samples + 2

        for i in range(2 ** samples):
            i_bin = np.array(list(np.binary_repr(i, width=samples)), dtype=int)
            j_bin = np.roll(i_bin, -1)  # Rotate left
            numt = np.sum(i_bin != j_bin)

            if numt <= 2:
                table[i] = np.sum(i_bin)
            else:
                table[i] = samples + 1

        return {'table': table, 'samples': samples, 'num': newMax}

    def getmapping(self, samples, mappingtype):
        """
        Returns the mapping structure for different LBP (Local Binary Patterns) types.

        Parameters:
        samples : int
            The number of samples (points) around the center pixel.
        mappingtype : str
            The type of mapping: 'u2', 'ri', or 'riu2'.

        Returns:
        mapping : dict
            A dictionary containing the 'table', 'samples', and 'num' (number of patterns).
        """

        if mappingtype == 'u2':
            return self.getmapping_u2(samples)
        elif mappingtype == 'ri':
            return self.getmapping_ri(samples)
        elif mappingtype == 'riu2':
            return self.getmapping_riu2(samples)
        else:
            raise ValueError("Unsupported mapping type")

    def get_ldzp(self):
        samples = 8
        mappingtype = 'u2'
        mapping = self.getmapping(samples, mappingtype)
        output = self.ZigZag(self.image, mapping, 'h')
        return output
