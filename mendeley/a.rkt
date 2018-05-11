#lang racket


(require db)
(require pict pdf-read)
(require "pdf-read-extra.rkt")

(define conn
  (sqlite3-connect #:database "/home/hebi/.local/share/data/Mendeley Ltd./Mendeley Desktop/lihebi.com@gmail.com@www.mendeley.com.sqlite"
                   #:mode 'read-only))

;; (query-rows conn "select * from Groups")
;; (query-rows conn "select * from FileHighlights")


(define query
  "SELECT Files.localUrl, FileHighlightRects.page,
                    FileHighlightRects.x1, FileHighlightRects.y1,
                    FileHighlightRects.x2, FileHighlightRects.y2,
                    FileHighlights.documentId,
Groups.name

            FROM Files
            LEFT JOIN FileHighlights
                ON FileHighlights.fileHash=Files.hash
            LEFT JOIN FileHighlightRects
                ON FileHighlightRects.highlightId=FileHighlights.id
LEFT JOIN RemoteDocuments
ON RemoteDocuments.documentId=FileHighlights.documentId
LEFT JOIN Groups
ON Groups.id=RemoteDocuments.groupId
            WHERE (FileHighlightRects.page IS NOT NULL)
"
)

(define (get-group-document-ids group-name)
  (let ([query (~a "SELECT documentId
            FROM RemoteDocuments
JOIN Groups
ON Groups.id=RemoteDocuments.groupId
WHERE Groups.name=\"" group-name "\"")])
    (map first (map vector->list (query-rows conn query)))))

;; (get-group-document-ids "nsf")

#;
(filter (λ (id)
          (non-empty-string? (get-document-file-from-id id)))
        (get-group-document-ids "nsf"))


(define (get-document-file id)
  "Return file name or empty string"
  (let ([query (~a "SELECT Files.localUrl from Files LEFT JOIN
DocumentFiles ON DocumentFiles.hash=Files.hash where
DocumentFiles.documentId=" id)])
    (let ([query-result (query-rows conn query)])
      (if (empty? query-result) ""
          (substring
           (vector-ref (first query-result) 0)
           (string-length "file://"))))))

(define (get-document-hash id)
  (let ([query (~a "SELECT Files.hash from Files LEFT JOIN
DocumentFiles ON DocumentFiles.hash=Files.hash where
DocumentFiles.documentId=" id)])
    (let ([query-result (query-rows conn query)])
      (if (empty? query-result) ""
          (vector-ref (first query-result) 0)))))

;; (get-document-hash-from-id 38)

(define (get-highlight-spec id)
  ;; note that there might be multiple id corresponding to the same
  ;; file. The highlights, however, relate to one id. So, using
  ;; document id to get highlights will not get all the highlights
  (let ([query (~a "select FileHighlightRects.page, 
FileHighlightRects.x1, FileHighlightRects.y1,
FileHighlightRects.x2, FileHighlightRects.y2
FROM Files
LEFT JOIN FileHighlights
ON FileHighlights.fileHash=Files.hash
LEFT JOIN FileHighlightRects
ON FileHighlightRects.highlightId=FileHighlights.id
where Files.hash=\"" (get-document-hash id) "\"
order by FileHighlightRects.page")])
    (query-rows conn query)
    ;; group by pages
    (group-by (λ (v) (first v))
              (map vector->list (query-rows conn query)))))

;; (get-highlight-spec 38)


;; 1. sort and partition highlights based on file and page
;; 2. for each page, render a pict, mark all highlight

;; Now, I want to get the whole text, and embed the highlight inside
;; with <hl></hl> tags.

;; Also, I want to get the font name and size.


(define (get-highlight-text id)
  (let ([file-hl (get-highlight-spec id)]
        [f (get-document-file id)])
    (for/list ([page-hl (in-list file-hl)])
      (let* ([p (sub1 (first (first page-hl)))]
             [pdf (pdf-page f p)]
             [view (page->pict pdf)])
        (for/list ([hl (in-list page-hl)])
          (match-let*
              ([(list x1 y1 x2 y2)
                (mendeley-rect->pdf-read-rect pdf (drop hl 1))])
            (page-text-in-rect pdf 'glyph x1 y2 x2 y1)))))))

(define (mendeley-rect->pdf-read-rect pdf rect)
  (match-let ([(list x1 y1 x2 y2) rect])
    (let ([height (second (page-size pdf))])
      (list x1 (- height y1) x2 (- height y2)))))

;; (get-highlight-text 38)

(define (visualize-highlight id)
  (let ([file-hl (get-highlight-spec id)]
        [f (get-document-file id)])
    (for/list ([page-hl (in-list file-hl)])
      (let* ([p (sub1 (first (first page-hl)))]
             [pdf (pdf-page f p)])
        (scale
         (for/fold ([view (page->pict pdf)])
                   ([hl (in-list page-hl)])
           (match-let*
               ([(list x1 y1 x2 y2)
                 (mendeley-rect->pdf-read-rect pdf (drop hl 1))])
             (pin-over view x1 y2
                       (cellophane
                        (colorize
                         (filled-rectangle (- x2 x1) (- y1 y2))
                         "yellow")
                        0.5))))
         1.5)))))

;; (visualize-highlight 38)


;; (page->pict "/home/hebi/Downloads/FSE_2018_paper_239.pdf")

(define dirichlet-pdf (pdf-page "/home/hebi/.local/share/data/Mendeley%20Ltd./Mendeley%20Desktop/Downloaded/Blei,%20Ng,%20Jordan%20-%202012%20-%20Latent%20Dirichlet%20Allocation.pdf"
                                0))

;; Alright, now we can get the font name and size. The font name
;; contians bold, italic. There is an attribute for is_underlined or
;; not.
(define (attr-test)
  (page-attr dirichlet-pdf))

;; (length (page-text-with-layout dirichlet-pdf))

(define (add-empty-attr text-with-layout)
  (map (λ (l) (append l '(())))))
(define (in-rect? small big)
  "Check if small is within big"
  (match-let ([(list sx1 sy1 sx2 sy2) small]
              [(list bx1 by1 bx2 by2) big])
    ;; FIXME logic
    (and (> sx1 bx1)
         (> sy1 by1)
         (< sx2 bx2)
         (< sy2 by2))))

(define (mark-hl attr-text hls)
  "Precondition: same page"
  (map (λ (letter)
         (if (in-rect? (second letter) hls)
             (append (drop-right letter 1)
                     (list (append (last letter) '(hl))))
             letter))
       attr-text))

(define (filter-hl attr-text)
  (filter (λ (letter)
            (member 'hl (last letter)))
          attr-text))

(define (page-text-not-in-rects page rects)
  "Return the text that is not in rects. Concatenate them.")

;; (page-find-text dirichlet-pdf "deployed in modern Internet ")

(define (search)
  (scale (let ([page dirichlet-pdf])
           (for/fold ([pageview (page->pict page)])
                     ([bounding-box
                       (in-list (page-find-text
                                 page
                                 "describe latent"))])
             (match-define (list x1 y1 x2 y2) bounding-box)
             (println (page-text-in-rect dirichlet-pdf 'glyph x1 y1 x2 y2))
             (println
              (page-attr-in-rect dirichlet-pdf x1 y1 x2 y2))
             ;; Each match's bounding box ^
             (pin-over pageview x1 y1
                       (cellophane
                        (colorize (filled-rectangle (- x2 x1) (- y2 y1)) "yellow")
                        0.5)))) 1.5))
