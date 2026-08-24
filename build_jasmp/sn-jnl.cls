%% ---------------------------------------------------------------------------
%% sn-jnl.cls -- LOCAL STRUCTURE-CHECK SHIM. NOT Springer Nature's class.
%%
%% Springer does not distribute the real sn-jnl.cls through CTAN, so it cannot
%% be installed locally. This shim emulates just enough of its author interface
%% (\fnm, \sur, \affil, \abstract, \keywords, \bmhead, starred \author) at
%% Springer's single-column text measure (375pt) to compile the manuscript and
%% catch real problems: overfull boxes, undefined references, broken floats,
%% unresolved citations.
%%
%% The resulting PDF is a CONTENT AND STRUCTURE check only. Line breaks, page
%% count, and all visual styling will differ from the real class. Build the
%% submission on Overleaf with Springer's template -- see README_JASMP.md.
%% ---------------------------------------------------------------------------
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{sn-jnl}[2026/08/24 structure-check shim, not Springer's class]

% Swallow the real class's options (sn-mathphys-num, pdflatex, sn-vancouver...).
\DeclareOption*{}
\ProcessOptions\relax
\LoadClass[10pt,onecolumn]{article}

\RequirePackage[a4paper,textwidth=375pt,textheight=634pt,centering]{geometry}
\RequirePackage{graphicx}
\RequirePackage{amsmath}
\RequirePackage[numbers,square,comma,sort&compress]{natbib}
\bibliographystyle{unsrtnat}   % numeric, closest available to sn-mathphys-num

% ---- address-field passthroughs -------------------------------------------
\newcommand{\fnm}[1]{#1}
\newcommand{\sur}[1]{#1}
\newcommand{\orgdiv}[1]{#1}
\newcommand{\orgname}[1]{#1}
\newcommand{\orgaddress}[1]{#1}
\newcommand{\city}[1]{#1}
\newcommand{\state}[1]{#1}
\newcommand{\country}[1]{#1}
\newcommand{\postcode}[1]{#1}
\newcommand{\street}[1]{#1}

% ---- author / affiliation collection --------------------------------------
\newcommand{\sn@authors}{}
\newcommand{\sn@affils}{}
\newcommand{\sn@emails}{}

% \author[1]{...} and \author*[2]{...}; the star marks the corresponding author.
\RenewDocumentCommand{\author}{s o +m}{%
  \g@addto@macro\sn@authors{#3\textsuperscript{\IfValueT{#2}{#2}\IfBooleanT{#1}{,$*$}}\quad}%
}
\NewDocumentCommand{\email}{+m}{%
  \g@addto@macro\sn@emails{\texttt{#1}\quad}%
}
\NewDocumentCommand{\affil}{o +m}{%
  \g@addto@macro\sn@affils{\textsuperscript{\IfValueT{#1}{#1}}#2\par}%
}

% \title[running head]{title} -- keep the full title, drop the running head.
% Set \@title directly rather than \let-ing the original: \let picks up the
% xparse wrapper, and calling it back recurses until TeX runs out of memory.
\RenewDocumentCommand{\title}{o +m}{\gdef\@title{#2}}

\RenewDocumentCommand{\abstract}{+m}{\gdef\sn@abstract{#1}}
\NewDocumentCommand{\keywords}{+m}{\gdef\sn@keywords{#1}}
\providecommand{\sn@abstract}{}
\providecommand{\sn@keywords}{}

\RenewDocumentCommand{\maketitle}{}{%
  \begin{center}
    {\LARGE\bfseries\@title\par}\vspace{1em}
    {\large\sn@authors\par}\vspace{0.6em}
    {\small\sn@affils}\vspace{0.4em}
    {\footnotesize\sn@emails\par}
  \end{center}
  \vspace{0.8em}
  \noindent\textbf{Abstract}\quad\sn@abstract\par\vspace{0.6em}
  \noindent\textbf{Keywords}\quad\sn@keywords\par\vspace{1.2em}
}

% ---- misc sn-jnl-only commands --------------------------------------------
\newcommand{\bmhead}[1]{\par\medskip\noindent\textbf{#1}\par\nobreak\smallskip}
\providecommand{\botrule}{}
\providecommand{\backmatter}{\par\bigskip}
\providecommand{\pacs}[2][]{}
