module Main where
import System.Environment

import Text.ParserCombinators.Parsec hiding (spaces)
import System.Environment

main :: IO ()
main = do
  -- args <- getArgs
  -- putStrLn ("Hello, " ++ args !! 0)
  -- (expr:_) <- getArgs
  putStrLn (readExpr "$")


myadd :: Num a => a -> a -> a
myadd a b = a + b

symbol :: Parser Char
symbol = oneOf "!#$%&|*+-/:<=>?@^_~"

readExpr :: String -> String
readExpr input = case parse symbol "lisp" input of
  Left err -> "No match: " ++ show err
  Right val -> "Found value"
