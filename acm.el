;; A lisp program to download ACM digital library proceeding entries automatically.

;; The interface:
;; this-script PLDI 2016
;; Then it outputs a bib file containing all entries of that proceeding

;; It should be configurable for the format of bib:
;; either the google scholar format, or the acm format

;; optionally I want to rename the entry keys to 2016-PLDI-Author format

;; Also I want to download the entries automatically


(defun acm-pdf-url (*doi-utils-redirect*)
  "Get a url to the pdf from *DOI-UTILS-REDIRECT* for ACM urls."
  (when (string-match "^http://dl.acm.org" *doi-utils-redirect*)
    (with-current-buffer (url-retrieve-synchronously *doi-utils-redirect*)
          (goto-char (point-min))
          (when (re-search-forward "<a name=\"FullTextPDF\".*href=\"\\([[:ascii:]]*?\\)\"")
            (concat "http://dl.acm.org/" (match-string 1))))))

(defun get-pldi-2016()
  "Get the 2016 pldi entries"
  ;; http://dl.acm.org/citation.cfm?id=2908080
  (let ((url "http://dl.acm.org/citation.cfm?id=2908080"))
    (with-current-buffer (url-retrieve-synchronously url)
      (goto-char (point-min))
      ;; (re-search-forward "<a href=\"citation\\.cfm\\?id=\\([[:digit:]]+\\))")
      (pwd)
      (write-file "a.html")
      ;; (match-string 1)
      )))



;; (mark-whole-buffer)
;; (let ((dom (libxml-parse-html-region (point-min) (point-max))))
;;   (prin1 dom)))))

;; a href="citation\.cfm\?id=\([[:digit:]]+\)
 
(get-pldi-2016)
