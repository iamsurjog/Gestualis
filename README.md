<h1 align="center">
<img src="https://github.com/iamsurjog/Gestualis/blob/main/logo/Logo.png" width="300">
</h1><br>

## <a href="#Installation">Installation</a>
## <a href="#Functionality">Functionality</a>
## <a href="#Usage">Usage</a>

# Installation
In order to get started, install gestualis with:

    pip install gestualis
or if there is no virtual environment being used then:

    py -m pip install gestualis

# Functionality
This module can:

- <a href="#storing">Capture gesture and store it in form of .dat files</a>
- <a href="#detection">Detect what a single gesture is closest to within the data which has been programmed by user</a>
- <a href="#livefeed">Detect gestures from the repertoire of data created by user to detect symbols on a live feed</a>

# Usage
<section id="storing">
To store a gesture

    from gestualis import HandDetector

    h = HandDetector()
    res = h.capture()
    h.store(res, "name_of_gesture")
</section>

<section id="detection">
To detect a gesture

    from gestualis import HandDetector

    h = HandDetector()
    res = h.capture()
    h.check(res)
</section>

<section id="livefeed">
To detect gestures from a live feed

    from gestualis import HandDetector

    h = HandDetector()
    res = h.livefeed()
    print(res)
</section>

