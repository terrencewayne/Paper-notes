# Why cv2.circle fails to draw after np.transpose?

A read image is C_contiguous in numpy by default, however when transposed, such as a = b.transpose(), the result a is F_contiguous. Because the default assignment is shallow copy, 
so a and b share the same storage and transpose in index, thus b's row converts to a's column, C_contiguous comes to F_contiguous.

Meanwhile, cv2.circle is supposed to receive a C_contiguous array. Therefore it fails.

we can use a = b.transpose().copy() to avoid the contiguous type.
