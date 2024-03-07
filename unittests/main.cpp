#include <iostream>
#include <gtest/gtest.h>
// CHECK GTEST ENVIRONMENT
int add(int a, int b)
{
    return a+b;
}

TEST(TestSample, TestAddition)
{
    ASSERT_EQ(3,add(1,2));
}

int main(int argc, char**argv)
{
    testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}
