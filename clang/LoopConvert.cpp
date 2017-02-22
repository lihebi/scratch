#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"

#include "llvm/Support/CommandLine.h"

// #include "clang/AST/ASTConsumer.h"
// #include "clang/CodeGen/ObjectFilePCHContainerOperations.h"
// #include "clang/Driver/Options.h"
// #include "clang/Frontend/ASTConsumers.h"
// #include "clang/Frontend/CompilerInstance.h"
// #include "clang/Rewrite/Frontend/FixItRewriter.h"
// #include "clang/Rewrite/Frontend/FrontendActions.h"
// #include "clang/StaticAnalyzer/Frontend/FrontendActions.h"
// #include "clang/Tooling/CommonOptionsParser.h"
// #include "clang/Tooling/Tooling.h"
// #include "llvm/ADT/STLExtras.h"
// #include "llvm/Option/OptTable.h"
// #include "llvm/Support/Path.h"
// #include "llvm/Support/Signals.h"
// #include "llvm/Support/TargetSelect.h"


using namespace clang::tooling;
using namespace llvm;

// Apply a custom category to all command-line options so that they are the
// only ones displayed.
static llvm::cl::OptionCategory MyToolCategory("my-tool options");

// CommonOptionsParser declares HelpMessage with a description of the common
// command-line options related to the compilation database and input files.
// It's nice to have this help message in all tools.
static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

// A help message for this specific tool can be added afterwards.
static cl::extrahelp MoreHelp("\nMore help text...");

int main(int argc, const char **argv) {
  CommonOptionsParser OptionsParser(argc, argv, MyToolCategory);
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());
  return Tool.run(new FrontendActionFactory<clang::SyntaxOnlyAction>().get());
}
