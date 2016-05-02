# Machine Learning by Andrew Ng
Assignments of Machine Learning course by Andrew Ng.<br /> 
Course Website: https://www.coursera.org/learn/machine-learning/home/welcome<br />
Course Wiki: https://share.coursera.org/wiki/index.php/ML:Main<br />
<br />

## Environment
- MacOS X El Capitan
- Octave 4.0.0_5 (via homebrew/science)
<br />
<br />

## Install Octave
- [Octave wiki](http://wiki.octave.org/Octave_for_MacOS_X#Homebrew)
<br />
<br />

## Possible Issue
#### 1. Plotting
```
[ERROR] 
nuplot> set terminal aqua enhanced title "Figure 1"  font "*,6.66667" dashlength 1
                      ^
         line 0: unknown or ambiguous terminal type; type just 'set terminal' for a list

WARNING: Plotting with an 'unknown' terminal.
No output will be generated. Please select a terminal with 'set terminal'.


[SOLUTION]
$ brew uninstall gnuplot
$ brew install gnu plot —with-x11
```

#### 2. Font
```
[ERROR]
warning: could not match any font: *-normal-normal-10


[SOLUTION] - http://stackoverflow.com/a/35250118
# should add to ~/.bash_profile
$ export FONTCONFIG_PATH=/opt/X11/lib/X11/fontconfig
```

#### 3. Black rectangle when `print`
```
[SOLUTION] - http://stackoverflow.com/a/29221808
brew switch gnuplot 4.6.6
```
