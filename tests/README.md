# How to use test scripts for `thompson_pulsed`

Some test scripts in this directory ask to read example data. This data is not stored on GitHub because the files can get pretty big, so instead it is stored on the Thompson Lab OneDrive under `[OneDrive]:/thompson_pulsed/example_data`. To run these scripts, copy the aforementioned folder into the `tests/` directory of your `thompson_pulsed` distribution, and then try running the script again. The file structure should look like this:

```
tests/
|	README.md
|	e3l_tests.py  
|	...
+	example_data/
|	+	e3l/
|	|	|	3L_Test_NS0.txt
|	|	|	...
|	|	...
```