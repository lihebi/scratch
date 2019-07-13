#lang racket

(require (for-syntax syntax/parse
                     racket/string)
         syntax/parse/define
         rackunit)

(provide r^define
         r^match
         r^range
         r^or
         r^seq
         r^*)

(struct r^ (reg-str func width)
  #:transparent)

(define-syntax (to-r^ stx)
  (syntax-parse stx
    ;; FIXME why :char is not working
    [(_ e:char)
     #'42]
    [(_ e)
     #'e]
    [(_ (f e ...))
     ;; TODO check whether f is valid
     #'(f (to-r^ e) ...)]))

(define-syntax (test-m stx)
  (syntax-parse stx
    [(_ e:char)
     #''c]
    [(_ e)
     #'42]
    [(_ e ...)
     #'(list (test-m e) ...)]))
;; (test-m 1 2 #\a)


(define-syntax (r^define stx)
  (syntax-parse stx
    [(_ regexp-id r)
     #'(define regexp-id (to-r^ r))]))

(define-syntax (r^match stx)
  (syntax-parse stx
    [(_ r string-expr)
     #'(my-match (to-r^ r)
                 string-expr)]))

;; (regexp-match "([a-d])" "ab")

(define (my-match r string-expr)
  ;; FIMXE should this be a function or macro?
  ;; 1. extract the str part of r^
  ;; 2. call racket's regexp-match
  ;; 3. call r^-func on the result and return
  #;
  (println (regexp-match
            (string-append "^" (r^-reg-str r) "$")
            string-expr))
  ;; (println (r^-width r))
  (let ([res (regexp-match
              (string-append "^" (r^-reg-str r) "$")
              string-expr)])
    (if (not res) #f
        ((r^-func r)
         (list->vector
          (rest res))
         ;; the r^-width seems to start from 1
         (sub1 (r^-width r))))))


(define-syntax (r^range stx)
  (syntax-parse stx
    [(_ a b)
     ;; (println #'a)
     ;; (println (syntax-e #'a))
     #`(r^ #,(string-append "(["
                            (string (syntax-e #'a))
                            "-"
                            (string (syntax-e #'b))
                            "])")
           ;; (format "([~a-~a])" #'a #'b)
           (λ (strs i)
             (string-ref (vector-ref strs i) 0))
           1)]))

#;
(r^match (r^or (r^range #\a #\d)
               (r^range #\w #\z))
         "b")

;; (regexp-match "(([a-d])|([w-z]))" "b")
;; (rest (regexp-match "([a-d])" "b"))

(define (or-impl strs rs i)
  (if (empty? rs) #f
      (if (vector-ref strs (+ i 1))
          ((r^-func (first rs)) strs (+ i 1))
          (or-impl strs (rest rs)
                (+ i (r^-width (first rs)))))))

(define-syntax (r^or stx)
  (syntax-parse stx
    [
     (_ r ...)
     #`(r^ (string-append
            "("
            (string-join (list (r^-reg-str r) ...) "|")
            ")")
           (λ (strs i)
             (or-impl strs (list r ...) i))
           #;
           (let ([r1^ (first (list r ...))]
                 [r2^ (second (list r ...))])
             (λ (strs i)
               (define p1 (+ i 1))
               (define p2 (+ i 1 (r^-width r1^)))
               (cond
                 [(vector-ref strs p1)
                  ((r^-func r1^) strs p1)]
                 [(vector-ref strs p2)
                  ((r^func r2^) strs p2)]
                 [else #f])))
           1)]))

(define (seq-impl strs rs i)
  (if (empty? rs) '()
      (if (not (vector-ref strs (+ i 1))) #f
          (cons ((r^-func (first rs)) strs (+ i 1))
                (seq-impl strs (rest rs)
                          (+ i (r^-width (first rs))))))))
;; (vector-ref (vector 1 2 3) 2)

#;
(r^match (r^seq (r^* (r^range #\a #\d) #\e)
                (r^range #\x #\z))
         "adz")
;; (r^match (r^* (r^range #\a #\c) #\d)
;;          "abic")

(define-syntax (r^seq stx)
  (syntax-parse stx
    [(_ r ...)
     #`(r^ (string-append
            "("
            (string-join (list (r^-reg-str r) ...) "")
            ")")
           
           (λ (strs i)
             (list->vector (seq-impl strs (list r ...) i)))
           1)]))


#;(regexp-match "((([a-d])*)([x-z]))" "adz")
#;
(check-equal? (r^match (r^seq (r^* (r^range #\a #\d) #\e)
                              (r^range #\x #\z))
                       "adz")
              (vector-immutable #\d #\z))


(define-syntax (r^* stx)
  (syntax-parse stx
    [(_ r default-expr)
     #`(r^ (string-append "(" (r^-reg-str r) "*)")
           (λ (strs i)
             (cond
               [(vector-ref strs (+ i 1))
                (
                 (r^-func r)
                 ;; vector-ref
                 strs (+ i 1))]
               [else default-expr]))
           1)]))

(expand)


;; (check-equal? (r^match (r^range #\a #\d) "a") #\a)
;; (check-equal? (r^match (r^range #\a #\d) "b") #\b)
;; (r^match (r^range #\a #\d) "a")

;; (r^range #\a #\d)
