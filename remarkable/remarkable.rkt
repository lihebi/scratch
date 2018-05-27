#lang racket

(require json)
(require racket/path)
(require pdf-read)
(require pict)
(require file/convertible)

(define (download-meta! metadir)
  ;; scp all metadata from 10.11.99.1
  (when (not (directory-exists? metadir))
    (make-directory metadir))
  (let ([scp-cmd (~a "scp -r root@10.11.99.1:~/.local/share/remarkable/xochitl/*.metadata "
                     metadir)])
    (system scp-cmd)))

(define (show-contents metadir)
  ;; show uuid, filename pairs
  (for/list ([f (in-directory metadir)])
    (let* ([uuid (path->string
                  (path-replace-extension
                   (file-name-from-path f) ""))]
           [meta (string->jsexpr (file->string f))]
           [filename (meta->filename meta)])
      (list uuid filename))))

(define (meta->filename meta)
  (hash-ref meta 'visibleName))

;; - get UUID.pdf
;; - get UUID.lines
(define (download-uuid uuid)
  (when (directory-exists? uuid)
    (delete-directory/files uuid))
  (make-directory uuid)
  (let ([scp-cmd (~a "scp root@10.11.99.1:~/.local/share/remarkable/xochitl/"
                     uuid "* " uuid)])
    (system scp-cmd)))

(define (get-pdfunite-cmds pdf-dir annotate-file)
  (letrec ([pdf-lst (sort (for/list ([p (in-directory pdf-dir)])
                            (path->string p))
                          (lambda (p1 p2)
                            (let ([get-id (λ (p)
                                            (string->number
                                             (second
                                              (regexp-match #px"([0-9]*)\\.pdf" p))))])
                              (< (get-id p1)
                                 (get-id p2)))))]
           [func (λ (lst out)
                   (if (<= (length lst) 100)
                       (list (~a "pdfunite " (string-join lst " ") " " out))
                       (let ([part1 (~a out "-part1.pdf")]
                             [part2 (~a out "-part2.pdf")])
                         (append (func (take lst 100) part1)
                                 (func (drop lst 100) part2)
                                 (list (~a "pdfunite " part1 " " part2 " " out))))))])
    (func pdf-lst (path->string annotate-file))))

(module+ test
  (for ([cmd (get-pdfunite-cmds "2a082b3c-9997-4b08-a7ae-9c3556ea5b92/pdf"
                                (string->path "tmp/annotate.pdf"))])
    (system cmd))

  (first (get-pdfunite-cmds "2a082b3c-9997-4b08-a7ae-9c3556ea5b92/pdf"
                            (string->path "tmp/annotate.pdf")))

  (sort (for/list ([p (in-directory "2a082b3c-9997-4b08-a7ae-9c3556ea5b92/pdf")])
          (path->string p))
        (lambda (p1 p2)
          (let ([get-id (λ (p)
                          (string->number
                           (second
                            (regexp-match #px"([0-9]*).pdf" p))))])
            (< (get-id p1)
               (get-id p2)))))

  

  (take (sort (directory-list "2a082b3c-9997-4b08-a7ae-9c3556ea5b92/pdf")
         (lambda (p1 p2)
           (let ([get-id (λ (p)
                           (string->number
                            (second
                             (regexp-match #px"([0-9]*).pdf" (path->string p)))))])
             (< (get-id p1)
                (get-id p2))))) 50)

  
  
  )

(define (save-file p filename)
  (let ([out (open-output-file filename
                               #:mode 'binary
                               #:exists 'replace)])
    (write-bytes (convert p 'pdf-bytes)
                 out)
    (close-output-port out)))


(define (annotate uuid)
  (let ([svg-dir (build-path uuid "svg")]
        [pdf-dir (build-path uuid "pdf")]
        [final-pdf-dir (build-path uuid "final-pdf")]
        [original-pdf (build-path uuid (~a uuid ".pdf"))]
        [lines-file (build-path uuid (~a uuid ".lines"))]
        [annotate-file (build-path uuid "annotate.pdf")]
        [final-file (build-path uuid "final.pdf")])
    (when (directory-exists? svg-dir)
      (delete-directory/files svg-dir))
    (make-directory svg-dir)
    (when (directory-exists? pdf-dir)
      (delete-directory/files pdf-dir))
    (make-directory pdf-dir)
    (when (directory-exists? final-pdf-dir)
      (delete-directory/files final-pdf-dir))
    (make-directory final-pdf-dir)
    ;; - rM2svg --coloured_annotations -i UUID.lines -o UUID
    (displayln "rM2svg")
    (let ([lines->svg-cmd (~a "rM2svg -i " (path->string lines-file)
                              " -o " (path->string (build-path svg-dir uuid)))])
      (system lines->svg-cmd))
    ;; - optional: transform
    ;; - for *.svg do rsvg-convert -f pdf -o XXX.pdf XXX.svg
    (displayln "rsvg-convert")
    (for ([p (directory-list svg-dir)])
      (let ([cmd 
             (~a "rsvg-convert -f pdf -o "
                 (path->string (build-path pdf-dir (path-replace-extension p ".pdf")))
                 " " (path->string (build-path svg-dir p)))])
        (system cmd)))
    (displayln "pdfunite")
    ;; - pdfunite *.pdf UUID_annot.pdf
    (for ([pdfunite-cmd (get-pdfunite-cmds pdf-dir annotate-file)])
      (system pdfunite-cmd))
    ;; - pdftk UUID.pdf multistamp UUID_annot.pdf output final.pdf
    ;; (displayln "pdftk")
    #;
    (let ([pdftk-cmd (~a "pdftk "
                         (path->string original-pdf)
                         " multistamp "
                         (path->string annotate-file)
                         " output " (path->string final-file))])
      (system pdftk-cmd))
    (displayln "stamp annotation")
    (stamp-annotate original-pdf annotate-file final-pdf-dir)
    ;; unite
    (displayln "final unite")
    (for ([pdfunite-cmd (get-pdfunite-cmds final-pdf-dir final-file)])
      (system pdfunite-cmd))))

(define (stamp-annotate original-file annotate-file outdir)
  (let ([ct (min (pdf-count-pages original-file)
                 (pdf-count-pages annotate-file))])
    (for ([i (range ct)])
      (display ".")
      (flush-output)
      (let ([original-pdf (page->pict (pdf-page original-file i))]
            [annotate-pdf (page->pict (pdf-page annotate-file i))]
            [outfile (build-path outdir (~a i ".pdf"))])
        (let ([pict (ct-superimpose original-pdf
                                    (scale-to-fit annotate-pdf original-pdf #:mode 'preserve/max))])
          (save-file pict (path->string outfile)))))))

(module+ test
  (download-meta! "output")
  (show-contents "output")

  
  
  (define uuid "0d618cf2-81fe-479a-bce7-908281a78a3b")
  (download-uuid uuid)
  (annotate uuid)

  (download-uuid "0f01c9ed-1f52-437d-9606-8608efec6c02")
  (annotate "0f01c9ed-1f52-437d-9606-8608efec6c02")

  (download-uuid "f8d4f0a4-2051-4045-8f2d-7785657a9c3c")

  (download-uuid "2a082b3c-9997-4b08-a7ae-9c3556ea5b92")
  (annotate "2a082b3c-9997-4b08-a7ae-9c3556ea5b92")

  (page-size (pdf-page "0d618cf2-81fe-479a-bce7-908281a78a3b/annotate.pdf" 1))
  (page-size (pdf-page "0d618cf2-81fe-479a-bce7-908281a78a3b/pdf/0d618cf2-81fe-479a-bce7-908281a78a3b_05.pdf" 0))

  (page-size (pdf-page "2a082b3c-9997-4b08-a7ae-9c3556ea5b92/2a082b3c-9997-4b08-a7ae-9c3556ea5b92.pdf" 0))

  (define annot-pict (page->pict (pdf-page "0d618cf2-81fe-479a-bce7-908281a78a3b/annotate.pdf" 7)))
  (define p5-pict (page->pict (pdf-page "0d618cf2-81fe-479a-bce7-908281a78a3b/0d618cf2-81fe-479a-bce7-908281a78a3b.pdf" 7)))

  ;; (define annot-pict (page->pict (pdf-page "tmp/annotate.pdf" 7)))
  (define original-pict (page->pict (pdf-page "tmp/original.pdf" 7)))
  (page-size (pdf-page "tmp/annotate.pdf" 7))
  (page-size (pdf-page "tmp/original.pdf" 7))

  (save-file (ct-superimpose annot-pict original-pict)
             "test.pdf")



  (page-size (pdf-page "0d618cf2-81fe-479a-bce7-908281a78a3b/0d618cf2-81fe-479a-bce7-908281a78a3b.pdf" 0))
  (page-size (pdf-page "0d618cf2-81fe-479a-bce7-908281a78a3b/annotate.pdf" 0))

  ;; (/ 1053.0 612)
  ;; (/ 1404 792.0)

  (pict-width annot-pict)
  (pict-height annot-pict)
  (pict-width p5-pict)
  (pict-height p5-pict)

  

  (ct-superimpose p5-pict
                  (scale-to-fit annot-pict p5-pict #:mode 'preserve/max))

  
  (page-size (pdf-page "0d618cf2-81fe-479a-bce7-908281a78a3b/0d618cf2-81fe-479a-bce7-908281a78a3b.pdf" 1))
  (/ 1404 612)
  (/ 1872 792)
  
  (define meta (string->jsexpr
                (file->string "output/fc991453-9476-404d-ae66-ed68bcb52dc2.metadata")))
  
  (path-replace-extension
   (file-name-from-path (string->path "output/fc991453-9476-404d-ae66-ed68bcb52dc2.metadata")) "")
  
  (meta->filename meta)
  )
