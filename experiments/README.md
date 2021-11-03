# Experiments

From https://abseil.io/docs/python/guides/flags:

Flags may be loaded from text files in addition to being specified on the commandline.

This means that you can throw any flags you don’t feel like typing into a file, listing one flag per line. For example:

```bash
--myflag=myvalue
--nomyboolean_flag
```